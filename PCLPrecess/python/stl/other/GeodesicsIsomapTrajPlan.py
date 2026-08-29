#!/usr/bin/env python3
"""Benchmark all-source graph Dijkstra against point-cloud Heat Method."""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra as cpu_dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
import trimesh

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import cupyx.scipy.sparse.csgraph as cpx_csgraph
    import cupyx.scipy.linalg as cpx_linalg
except Exception:
    cp = None
    cpx_sparse = None
    cpx_csgraph = None
    cpx_linalg = None


# ------------------------------ easy-to-edit parameters ------------------------------
SAMPLE_DENSITY = 1.0
MAX_POINTS = 2000
K_NEIGHBORS = 12
HEAT_TIME_SCALE = 1.0
SOURCE_COUNT = 0
REPEAT_COUNT = 5
RANDOM_SEED = 0
PREFER_GPU = False
GPU_BATCH_SIZE = 64

EPSILON = np.finfo(np.float64).eps


PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "int64": "i8", "uint64": "u8", "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def select_file() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select STL or point-cloud file",
        filetypes=[
            ("Supported point cloud files", "*.stl *.ply *.csv"),
            ("STL files", "*.stl"),
            ("PLY files", "*.ply"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def finite_xyz(points: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{label} must contain XYZ columns")
    points = points[:, :3]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError(f"{label} contains fewer than three finite XYZ points")
    return points


def cap_points(points: np.ndarray) -> np.ndarray:
    if MAX_POINTS <= 0 or len(points) <= MAX_POINTS:
        return points
    rng = np.random.default_rng(RANDOM_SEED)
    selected = np.sort(rng.choice(len(points), MAX_POINTS, replace=False))
    return points[selected]


def load_stl(path: Path) -> np.ndarray:
    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No triangle mesh found in STL")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("STL does not contain a non-empty triangle mesh")
    requested = max(3, int(round(float(loaded.area) * SAMPLE_DENSITY)))
    count = min(requested, MAX_POINTS) if MAX_POINTS > 0 else requested
    points, _ = trimesh.sample.sample_surface(loaded, count, seed=RANDOM_SEED)
    return finite_xyz(points, "STL samples")


def ply_dtype(name: str, endian: str) -> np.dtype:
    if name.lower() not in PLY_DTYPES:
        raise ValueError(f"Unsupported PLY type: {name}")
    return np.dtype(endian + PLY_DTYPES[name.lower()])


def load_ply(path: Path) -> np.ndarray:
    """Read only scalar vertex properties, ignoring face/camera/range_grid."""
    with path.open("rb") as stream:
        if stream.readline().strip().lower() != b"ply":
            raise ValueError("Not a PLY file")
        fmt = ""
        elements: list[dict] = []
        current = None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("Unterminated PLY header")
            words = raw.decode("ascii", errors="replace").strip().split()
            if not words:
                continue
            key = words[0].lower()
            if key == "end_header":
                break
            if key == "format":
                fmt = words[1].lower()
            elif key == "element":
                current = {"name": words[1].lower(), "count": int(words[2]), "props": []}
                elements.append(current)
            elif key == "property" and current is not None:
                if len(words) == 3:
                    current["props"].append((words[1].lower(), words[2].lower()))
                elif words[1].lower() == "list":
                    current["props"].append(("list", words[-1].lower()))
        vertex_index = next((i for i, e in enumerate(elements) if e["name"] == "vertex"), None)
        if vertex_index is None:
            raise ValueError("PLY has no vertex element")
        if vertex_index != 0 and any(e["count"] for e in elements[:vertex_index]):
            raise ValueError("PLY vertex must be the first non-empty data element")
        vertex = elements[vertex_index]
        if any(p[0] == "list" for p in vertex["props"]):
            raise ValueError("List-valued vertex properties are unsupported")
        names = [p[1] for p in vertex["props"]]
        try:
            xyz = [names.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError("PLY vertices do not contain x, y, z") from exc
        if fmt == "ascii":
            data = np.empty((vertex["count"], 3), dtype=np.float64)
            for row in range(vertex["count"]):
                values = stream.readline().split()
                data[row] = [float(values[column]) for column in xyz]
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(f"f{i}", ply_dtype(p[0], endian)) for i, p in enumerate(vertex["props"])])
            raw = np.fromfile(stream, dtype=dtype, count=vertex["count"])
            if len(raw) != vertex["count"]:
                raise ValueError("Truncated binary PLY")
            data = np.column_stack([raw[f"f{i}"].astype(np.float64) for i in xyz])
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    return finite_xyz(data, "PLY")


def load_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("CSV is empty")
    header = [cell.strip().lower() for cell in rows[0]]
    if all(axis in header for axis in ("x", "y", "z")):
        indices = [header.index(axis) for axis in ("x", "y", "z")]
        body = rows[1:]
    else:
        try:
            [float(value) for value in rows[0][:3]]
        except (ValueError, IndexError) as exc:
            raise ValueError("CSV header must contain x, y, z") from exc
        indices = [0, 1, 2]
        body = rows
    return finite_xyz([[float(row[i]) for i in indices] for row in body], "CSV")


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        points = load_stl(path)
    elif suffix == ".ply":
        points = load_ply(path)
    elif suffix == ".csv":
        points = load_csv(path)
    else:
        raise ValueError("Supported extensions are STL, PLY and CSV")
    return cap_points(points)


def build_connected_knn(points: np.ndarray) -> tuple[sp.csr_matrix, int]:
    n = len(points)
    tree = cKDTree(points)
    k = min(max(2, K_NEIGHBORS), n - 1)
    while True:
        distances, indices = tree.query(points, k=k + 1, workers=-1)
        rows = np.repeat(np.arange(n), k)
        graph = sp.coo_matrix(
            (distances[:, 1:].reshape(-1), (rows, indices[:, 1:].reshape(-1))),
            shape=(n, n), dtype=np.float64,
        ).tocsr()
        graph = graph.maximum(graph.T)
        graph.eliminate_zeros()
        components, _ = connected_components(graph, directed=False)
        if components == 1:
            return graph, k
        if k == n - 1:
            raise ValueError("Unable to create a connected kNN graph")
        new_k = min(n - 1, max(k + 1, int(np.ceil(k * 1.25))))
        print(f"kNN graph has {components} components; increasing k: {k} -> {new_k}")
        k = new_k


def gpu_details() -> tuple[bool, str]:
    if cp is None or not PREFER_GPU:
        return False, "disabled" if not PREFER_GPU else "CuPy unavailable"
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        cuda = cp.cuda.runtime.runtimeGetVersion()
        return True, f"{name}; CUDA runtime {cuda}; CuPy {cp.__version__}"
    except Exception as exc:
        return False, f"CUDA unavailable: {exc}"


def synchronize(use_gpu: bool) -> None:
    if use_gpu:
        cp.cuda.Stream.null.synchronize()


def describe_times(name: str, values: list[float]) -> None:
    print(
        f"  {name}: mean={np.mean(values):.6f}s, median={np.median(values):.6f}s, "
        f"std={np.std(values):.6f}s, min={np.min(values):.6f}s, max={np.max(values):.6f}s"
    )


def benchmark_dijkstra(graph: sp.csr_matrix, sources: np.ndarray, gpu_ok: bool):
    use_gpu = bool(gpu_ok and hasattr(cpx_csgraph, "dijkstra"))
    if gpu_ok and not use_gpu:
        print("GPU Dijkstra is unavailable in this CuPy version: CPU fallback")
    start = time.perf_counter()
    gpu_graph = cpx_sparse.csr_matrix(graph) if use_gpu else None
    synchronize(use_gpu)
    preparation = time.perf_counter() - start

    def compute() -> np.ndarray:
        result = np.empty((len(sources), graph.shape[0]), dtype=np.float64)
        for begin in range(0, len(sources), max(1, GPU_BATCH_SIZE)):
            batch = sources[begin:begin + max(1, GPU_BATCH_SIZE)]
            if use_gpu:
                value = cpx_csgraph.dijkstra(gpu_graph, directed=False, indices=cp.asarray(batch))
                result[begin:begin + len(batch)] = cp.asnumpy(value)
            else:
                result[begin:begin + len(batch)] = cpu_dijkstra(
                    graph, directed=False, indices=batch
                )
        return result

    compute()  # warm-up
    core, total, output = [], [], None
    for _ in range(max(1, REPEAT_COUNT)):
        synchronize(use_gpu)
        begin = time.perf_counter()
        output = compute()
        synchronize(use_gpu)
        elapsed = time.perf_counter() - begin
        core.append(elapsed)
        total.append(preparation + elapsed)
    backend = "GPU" if use_gpu else ("CPU (configured)" if not PREFER_GPU else "CPU fallback")
    return output, preparation, core, total, backend


def build_heat_operators(points: np.ndarray, graph: sp.csr_matrix):
    n = len(points)
    edge_lengths = graph.data
    h = float(np.median(edge_lengths))
    rows, cols, gx_data, gy_data, gz_data = [], [], [], [], []
    masses = np.empty(n, dtype=np.float64)
    for i in range(n):
        neighbors = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
        delta = points[neighbors] - points[i]
        covariance = delta.T @ delta / max(1, len(neighbors))
        _, basis = np.linalg.eigh(covariance)
        tangent = basis[:, -2:]
        local = delta @ tangent
        length = np.linalg.norm(delta, axis=1)
        weights = np.exp(-np.square(length / max(h, EPSILON)))
        normal = local.T @ (weights[:, None] * local)
        normal += 1.0e-10 * max(float(np.trace(normal)), 1.0) * np.eye(2)
        coefficients = tangent @ np.linalg.solve(normal, (local.T * weights))
        center_coeff = -coefficients.sum(axis=1)
        for component, store in enumerate((gx_data, gy_data, gz_data)):
            rows.extend([i] * (len(neighbors) + 1)) if component == 0 else None
            cols.extend([*neighbors, i]) if component == 0 else None
            store.extend([*coefficients[component], center_coeff[component]])
        radius = max(float(np.median(length)), EPSILON)
        masses[i] = np.pi * radius * radius
    gx = sp.coo_matrix((gx_data, (rows, cols)), shape=(n, n)).tocsr()
    gy = sp.coo_matrix((gy_data, (rows, cols)), shape=(n, n)).tocsr()
    gz = sp.coo_matrix((gz_data, (rows, cols)), shape=(n, n)).tocsr()
    mass = sp.diags(masses, format="csr")
    laplacian = gx.T @ mass @ gx + gy.T @ mass @ gy + gz.T @ mass @ gz
    laplacian = (0.5 * (laplacian + laplacian.T)).tocsr()
    t = HEAT_TIME_SCALE * h * h
    heat_matrix = (mass + t * laplacian).tocsr()
    scale = max(float(laplacian.diagonal().mean()), EPSILON)
    regularization = 1.0e-8 * scale / max(float(masses.mean()), EPSILON)
    poisson_matrix = (laplacian + regularization * mass).tocsr()
    return gx, gy, gz, masses, heat_matrix, poisson_matrix, h


def solve_gpu_cholesky(factor, rhs):
    intermediate = cpx_linalg.solve_triangular(
        factor, rhs, lower=True, check_finite=False
    )
    return cpx_linalg.solve_triangular(
        factor.T, intermediate, lower=False, check_finite=False
    )


def prepare_heat(points: np.ndarray, graph: sp.csr_matrix, use_gpu: bool):
    start = time.perf_counter()
    gx, gy, gz, masses, heat_a, poisson_a, h = build_heat_operators(points, graph)
    if use_gpu:
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.set_pinned_memory_allocator(None)
        heat_dense = cpx_sparse.csr_matrix(heat_a).toarray()
        poisson_dense = cpx_sparse.csr_matrix(poisson_a).toarray()
        operators = {
            "gx": cpx_sparse.csr_matrix(gx), "gy": cpx_sparse.csr_matrix(gy),
            "gz": cpx_sparse.csr_matrix(gz),
            "m": cp.asarray(masses, dtype=cp.float64, blocking=True),
            "heat_factor": cp.linalg.cholesky(heat_dense),
            "poisson_factor": cp.linalg.cholesky(poisson_dense),
            "gpu": True, "h": h,
        }
        del heat_dense, poisson_dense
        synchronize(True)
    else:
        operators = {
            "gx": gx, "gy": gy, "gz": gz, "m": masses,
            "heat_lu": splu(heat_a.tocsc()),
            "poisson_lu": splu(poisson_a.tocsc()),
            "gpu": False, "h": h,
        }
    return operators, time.perf_counter() - start


def heat_compute(operators: dict, sources: np.ndarray) -> np.ndarray:
    n, batch_size = len(operators["m"]), max(1, GPU_BATCH_SIZE)
    result = np.empty((len(sources), n), dtype=np.float64)
    gpu = operators["gpu"]
    for begin in range(0, len(sources), batch_size):
        source_batch = sources[begin:begin + batch_size]
        if gpu:
            source_gpu = cp.asarray(source_batch)
            rhs = cp.zeros((n, len(source_batch)), dtype=cp.float64)
            rhs[source_gpu, cp.arange(len(source_batch))] = operators["m"][source_gpu]
            u = solve_gpu_cholesky(operators["heat_factor"], rhs)
            gradients = [operators[key] @ u for key in ("gx", "gy", "gz")]
            norm = cp.sqrt(sum(value * value for value in gradients))
            fields = [-value / cp.maximum(norm, EPSILON) for value in gradients]
            div = -sum(
                operators[key].T @ (operators["m"][:, None] * field)
                for key, field in zip(("gx", "gy", "gz"), fields)
            )
            phi = solve_gpu_cholesky(operators["poisson_factor"], div)
            values = cp.asnumpy(phi.T)
        else:
            rhs = np.zeros((n, len(source_batch)), dtype=np.float64)
            rhs[source_batch, np.arange(len(source_batch))] = operators["m"][source_batch]
            u = operators["heat_lu"].solve(rhs)
            gradients = [operators[key] @ u for key in ("gx", "gy", "gz")]
            norm = np.sqrt(sum(value * value for value in gradients))
            fields = [-value / np.maximum(norm, EPSILON) for value in gradients]
            div = -sum(
                operators[key].T @ (operators["m"][:, None] * field)
                for key, field in zip(("gx", "gy", "gz"), fields)
            )
            values = operators["poisson_lu"].solve(div).T
        values -= values[np.arange(len(source_batch)), source_batch][:, None]
        signs = np.where(np.median(values, axis=1) < 0.0, -1.0, 1.0)
        values *= signs[:, None]
        result[begin:begin + len(source_batch)] = np.maximum(values, 0.0)
    if len(sources) == n and np.array_equal(sources, np.arange(n)):
        result = 0.5 * (result + result.T)
        np.fill_diagonal(result, 0.0)
    return result


def benchmark_heat(points, graph, sources, gpu_ok):
    use_gpu = bool(gpu_ok)
    try:
        operators, preparation = prepare_heat(points, graph, use_gpu)
        heat_compute(operators, sources)  # warm-up
    except Exception as exc:
        if not use_gpu:
            raise
        print(f"Heat Method GPU preparation/warm-up failed ({type(exc).__name__}: {exc})")
        print("Heat Method backend: CPU fallback")
        use_gpu = False
        operators, cpu_preparation = prepare_heat(points, graph, False)
        preparation += cpu_preparation
        heat_compute(operators, sources)
    core, total, output = [], [], None
    for _ in range(max(1, REPEAT_COUNT)):
        synchronize(use_gpu)
        start = time.perf_counter()
        output = heat_compute(operators, sources)
        synchronize(use_gpu)
        elapsed = time.perf_counter() - start
        core.append(elapsed)
        total.append(preparation + elapsed)
    backend = "GPU reused Cholesky" if use_gpu else (
        "CPU sparse LU (configured)" if not PREFER_GPU else "CPU fallback"
    )
    return output, preparation, core, total, backend, operators["h"]


def validate_distances(name: str, distance: np.ndarray, sources: np.ndarray) -> None:
    diagonal = distance[np.arange(len(sources)), sources]
    print(
        f"{name} validation: NaN={np.isnan(distance).sum()}, inf={np.isinf(distance).sum()}, "
        f"negative={(distance < 0).sum()}, max source-self={np.max(np.abs(diagonal)):.6g}"
    )
    if np.any(~np.isfinite(distance)):
        raise ValueError(f"{name} contains non-finite distances")


def compare_distances(reference, heat, sources):
    finite = np.isfinite(reference) & np.isfinite(heat)
    a, b = reference[finite], heat[finite]
    difference = b - a
    metrics = {
        "MAE": float(np.mean(np.abs(difference))),
        "RMSE": float(np.sqrt(np.mean(difference * difference))),
        "Max absolute error": float(np.max(np.abs(difference))),
        "Relative Frobenius error": float(np.linalg.norm(difference) / max(np.linalg.norm(a), EPSILON)),
        "Pearson correlation": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan,
        "NaN count": int(np.isnan(heat).sum()), "inf count": int(np.isinf(heat).sum()),
        "negative count": int((heat < 0).sum()),
        "Diagonal error": float(np.max(np.abs(heat[np.arange(len(sources)), sources]))),
    }
    if heat.shape[0] == heat.shape[1] and np.array_equal(sources, np.arange(heat.shape[0])):
        metrics["Symmetry error"] = float(
            np.linalg.norm(heat - heat.T) / max(np.linalg.norm(heat), EPSILON)
        )
    else:
        metrics["Symmetry error"] = np.nan
    return metrics


def show_results(points, sources, dijkstra_distance, heat_distance, dijkstra_core, heat_core, metrics):
    source = sources[0]
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot1 = ax1.scatter(*points.T, c=dijkstra_distance[0], s=8, cmap="viridis")
    ax1.scatter(*points[source], c="red", s=60, marker="*")
    ax1.set_title("kNN Dijkstra distance")
    fig.colorbar(plot1, ax=ax1, shrink=0.65)
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot2 = ax2.scatter(*points.T, c=heat_distance[0], s=8, cmap="viridis")
    ax2.scatter(*points[source], c="red", s=60, marker="*")
    ax2.set_title("Point-cloud Heat Method distance")
    fig.colorbar(plot2, ax=ax2, shrink=0.65)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(dijkstra_distance.ravel(), heat_distance.ravel(), s=3, alpha=0.25)
    limit = max(float(dijkstra_distance.max()), float(heat_distance.max()))
    ax3.plot([0, limit], [0, limit], "r--", lw=1)
    ax3.set(xlabel="Dijkstra", ylabel="Heat", title=f"Distance comparison (r={metrics['Pearson correlation']:.4f})")
    ax3.grid(alpha=0.25)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(["Dijkstra", "Heat Method"], [np.median(dijkstra_core), np.median(heat_core)], color=["tab:blue", "tab:orange"])
    ax4.set_ylabel("Median core time (s)")
    ax4.set_title("Core computation speed")
    ax4.grid(axis="y", alpha=0.25)
    plt.show()


def main() -> None:
    path = select_file()
    if path is None:
        print("File selection cancelled.")
        return
    read_start = time.perf_counter()
    points = load_points(path)
    read_time = time.perf_counter() - read_start
    graph_start = time.perf_counter()
    graph, final_k = build_connected_knn(points)
    graph_time = time.perf_counter() - graph_start
    rng = np.random.default_rng(RANDOM_SEED)
    if SOURCE_COUNT <= 0 or SOURCE_COUNT >= len(points):
        sources = np.arange(len(points), dtype=np.int64)
    else:
        sources = np.sort(rng.choice(len(points), SOURCE_COUNT, replace=False))
    gpu_ok, gpu_text = gpu_details()

    print("\n=== Input and common graph ===")
    print(f"File: {path.resolve()}")
    print(f"File loading/sampling time (excluded): {read_time:.6f}s")
    print(f"Points N: {len(points)}")
    print(f"Undirected edges E: {graph.nnz // 2}")
    print(f"Final k: {final_k}")
    print(f"Sources S: {len(sources)}")
    print(f"Common graph preparation time: {graph_time:.6f}s")
    print(f"GPU: {gpu_text}")

    print("\n=== 1. kNN graph all/source-set Dijkstra ===")
    d_dist, d_prep, d_core, d_total, d_backend = benchmark_dijkstra(graph, sources, gpu_ok)
    validate_distances("Dijkstra", d_dist, sources)
    print(f"Backend: {d_backend}")
    print(f"Preparation: {d_prep:.6f}s")
    describe_times("Core", d_core)
    describe_times("Total (preparation + core)", d_total)

    print("\n=== 2. Point-cloud Geodesics in Heat ===")
    h_dist, h_prep, h_core, h_total, h_backend, h = benchmark_heat(points, graph, sources, gpu_ok)
    validate_distances("Heat Method", h_dist, sources)
    print(f"Backend: {h_backend}")
    print(f"Median edge length h: {h:.6g}; heat time t: {HEAT_TIME_SCALE * h * h:.6g}")
    print(f"Preparation (PCA/operators/solver): {h_prep:.6f}s")
    describe_times("Core", h_core)
    describe_times("Total (preparation + core)", h_total)

    speedup = float(np.median(d_core) / max(np.median(h_core), EPSILON))
    metrics = compare_distances(d_dist, h_dist, sources)
    print("\n=== Speed and correctness ===")
    print(f"Speedup (Dijkstra median core / Heat median core): {speedup:.6f}x")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    show_results(points, sources, d_dist, h_dist, d_core, h_core, metrics)


if __name__ == "__main__":
    main()
