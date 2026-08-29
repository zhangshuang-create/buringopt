#!/usr/bin/env python3
"""
UMAP UV visualization for STL meshes.

Run directly:
    py UMAP.py

The script opens a file dialog, loads an STL/OBJ/PLY mesh as a 3D point set,
maps the sampled 3D points to a 2D UV coordinate system using UMAP, then shows
the 3D model and the UV embedding side by side.

If umap-learn is installed, it is used. Otherwise, this file runs a lightweight
UMAP implementation: kNN graph -> fuzzy simplicial set -> 2D optimization.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


EPS = 1.0e-12


@dataclass
class PointSource:
    points: np.ndarray
    mesh: Optional[trimesh.Trimesh]
    source_indices: np.ndarray
    name: str


@dataclass
class UMAPResult:
    points: np.ndarray
    uv: np.ndarray
    source_indices: np.ndarray
    graph_edges: np.ndarray
    graph_weights: np.ndarray
    method: str
    original_count: int
    sampled_count: int


def select_model_file_dialog() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("tkinter is unavailable, cannot open file dialog.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select STL mesh for UMAP UV",
        filetypes=[
            ("Mesh files", "*.stl *.obj *.ply"),
            ("STL files", "*.stl"),
            ("OBJ files", "*.obj"),
            ("PLY files", "*.ply"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def load_mesh_points(path: str | Path) -> PointSource:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {path}.")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh {path} has no vertices.")

    mesh = mesh.copy()
    mesh.merge_vertices()
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    points = np.asarray(mesh.vertices, dtype=float)
    return PointSource(points=points, mesh=mesh, source_indices=np.arange(len(points)), name=path.name)


def deterministic_sample(source: PointSource, max_points: int) -> PointSource:
    points = source.points
    if max_points <= 0 or len(points) <= max_points:
        return source

    selected = np.empty(int(max_points), dtype=np.int64)
    center = points.mean(axis=0)
    selected[0] = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    min_dist2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, int(max_points)):
        selected[i] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    selected.sort()

    return PointSource(
        points=points[selected],
        mesh=source.mesh,
        source_indices=source.source_indices[selected],
        name=source.name,
    )


def standardize_points(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=float)
    out = out - out.mean(axis=0)
    scale = np.std(out, axis=0)
    scale[scale < EPS] = 1.0
    return out / scale


def normalize_uv(uv: np.ndarray) -> np.ndarray:
    out = np.asarray(uv, dtype=float).copy()
    span = np.ptp(out, axis=0)
    scale = max(float(np.max(span)), EPS)
    out = out / scale
    out -= out[0]
    return out


def smooth_knn_dist(distances: np.ndarray, k: int, local_connectivity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    target = math.log2(float(k))
    n_samples = distances.shape[0]
    rhos = np.zeros(n_samples, dtype=float)
    sigmas = np.zeros(n_samples, dtype=float)

    def membership_sum(row: np.ndarray, rho: float, sigma: float) -> float:
        total = 0.0
        for d in row:
            if d - rho <= 0.0:
                total += 1.0
            else:
                total += math.exp(-(d - rho) / sigma)
        return total

    for i in range(n_samples):
        row = distances[i]
        non_zero = row[row > EPS]
        if len(non_zero) == 0:
            rhos[i] = 0.0
        elif local_connectivity <= 1.0:
            rhos[i] = float(non_zero[0])
        else:
            index = int(math.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0 and index < len(non_zero):
                rhos[i] = float(non_zero[index - 1])
                if interpolation > EPS:
                    rhos[i] += interpolation * float(non_zero[index] - non_zero[index - 1])
            else:
                rhos[i] = float(non_zero[-1])

        hi = 1.0
        while membership_sum(row, rhos[i], hi) < target:
            hi *= 2.0
            if hi > 1.0e6:
                break
        lo = 0.0
        mid = hi
        for _ in range(64):
            mid = (lo + hi) / 2.0
            psum = 0.0
            for d in row:
                if d - rhos[i] <= 0.0:
                    psum += 1.0
                else:
                    psum += math.exp(-(d - rhos[i]) / mid)
            if abs(psum - target) < 1.0e-5:
                break
            if psum > target:
                hi = mid
            else:
                lo = mid
        sigmas[i] = max(mid, EPS)
    return sigmas, rhos


def fuzzy_simplicial_set(points: np.ndarray, n_neighbors: int) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    neighbor_count = min(max(2, int(n_neighbors)), len(points) - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=neighbor_count + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    sigmas, rhos = smooth_knn_dist(distances, neighbor_count)
    rows = []
    cols = []
    vals = []
    for i in range(len(points)):
        for j, d in zip(indices[i], distances[i]):
            if d - rhos[i] <= 0.0:
                value = 1.0
            else:
                value = math.exp(-(d - rhos[i]) / sigmas[i])
            rows.append(i)
            cols.append(int(j))
            vals.append(float(value))

    directed = coo_matrix((vals, (rows, cols)), shape=(len(points), len(points))).tocsr()
    graph = directed + directed.T - directed.multiply(directed.T)
    graph = graph.tocoo()
    mask = graph.row != graph.col
    graph = coo_matrix((graph.data[mask], (graph.row[mask], graph.col[mask])), shape=graph.shape).tocsr()
    graph.eliminate_zeros()
    edges = np.column_stack(graph.nonzero()).astype(np.int64)
    weights = np.asarray(graph[edges[:, 0], edges[:, 1]]).reshape(-1).astype(float)
    return graph, edges, weights


def pca_initialization(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < 2:
        return np.column_stack([centered[:, 0], np.zeros(len(centered))])
    return centered @ vt[:2].T


def spectral_initialization(graph: csr_matrix, points: np.ndarray, random_state: int) -> np.ndarray:
    try:
        weights = np.asarray(graph.sum(axis=1)).reshape(-1)
        laplacian = diags(weights) - graph
        values, vectors = eigsh(laplacian, k=3, which="SM")
        order = np.argsort(values)
        embedding = vectors[:, order[1:3]]
        rng = np.random.default_rng(random_state)
        embedding += rng.normal(scale=1.0e-4, size=embedding.shape)
        return embedding
    except Exception:
        return pca_initialization(points)


def find_ab_params(spread: float, min_dist: float) -> Tuple[float, float]:
    def curve(x, a, b):
        return 1.0 / (1.0 + a * np.power(x, 2.0 * b))

    x = np.linspace(0.0, spread * 3.0, 300)
    y = np.ones_like(x)
    mask = x > min_dist
    y[mask] = np.exp(-(x[mask] - min_dist) / spread)
    try:
        params, _ = curve_fit(curve, x, y, p0=(1.0, 1.0), bounds=(0.0, np.inf), maxfev=10000)
        return float(params[0]), float(params[1])
    except Exception:
        return 1.5769434603113077, 0.8950608781227859


def optimize_umap_embedding(
    points: np.ndarray,
    graph: csr_matrix,
    edges: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    learning_rate: float,
    min_dist: float,
    spread: float,
    negative_sample_rate: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    uv = spectral_initialization(graph, points, random_state)
    uv = normalize_uv(uv)
    uv += rng.normal(scale=0.0001, size=uv.shape)

    if len(edges) == 0:
        return uv

    a, b = find_ab_params(spread, min_dist)
    weights = np.clip(weights, 0.0, 1.0)
    probabilities = weights / max(float(np.sum(weights)), EPS)
    samples_per_epoch = min(len(edges), max(1000, len(points) * 4))
    negative_sample_rate = max(1, int(negative_sample_rate))

    for epoch in range(max(1, int(n_epochs))):
        alpha = float(learning_rate) * (1.0 - epoch / max(1, int(n_epochs)))
        chosen = rng.choice(len(edges), size=samples_per_epoch, replace=True, p=probabilities)
        for edge_index in chosen:
            i, j = edges[int(edge_index)]
            diff = uv[i] - uv[j]
            dist2 = max(float(np.dot(diff, diff)), EPS)
            pow_dist = dist2 ** b
            grad_coeff = -2.0 * a * b * (dist2 ** (b - 1.0)) / (1.0 + a * pow_dist)
            grad = np.clip(grad_coeff * diff, -4.0, 4.0) * alpha
            uv[i] += grad
            uv[j] -= grad

            for _ in range(negative_sample_rate):
                k = int(rng.integers(0, len(points)))
                if k == i:
                    continue
                neg_diff = uv[i] - uv[k]
                neg_dist2 = max(float(np.dot(neg_diff, neg_diff)), EPS)
                neg_pow = neg_dist2 ** b
                neg_coeff = 2.0 * b / ((0.001 + neg_dist2) * (1.0 + a * neg_pow))
                neg_grad = np.clip(neg_coeff * neg_diff, -4.0, 4.0) * alpha
                uv[i] += neg_grad
    return normalize_uv(uv)


def compute_umap_uv(
    source: PointSource,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    n_epochs: int,
    learning_rate: float,
    negative_sample_rate: int,
    random_state: int,
    force_builtin: bool = False,
) -> UMAPResult:
    points_scaled = standardize_points(source.points)

    try:
        if force_builtin:
            raise ImportError("forced built-in UMAP fallback")
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            n_components=2,
            random_state=random_state,
        )
        uv = normalize_uv(reducer.fit_transform(points_scaled))
        graph = reducer.graph_.tocsr()
        edges = np.column_stack(graph.nonzero()).astype(np.int64)
        weights = np.asarray(graph[edges[:, 0], edges[:, 1]]).reshape(-1).astype(float)
        method = "umap-learn"
    except Exception:
        graph, edges, weights = fuzzy_simplicial_set(points_scaled, n_neighbors)
        uv = optimize_umap_embedding(
            points_scaled,
            graph,
            edges,
            weights,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            random_state=random_state,
        )
        method = "built-in lightweight UMAP"

    return UMAPResult(
        points=source.points,
        uv=uv,
        source_indices=source.source_indices,
        graph_edges=edges,
        graph_weights=weights,
        method=method,
        original_count=len(source.points),
        sampled_count=len(source.points),
    )


def export_uv_csv(path: str | Path, result: UMAPResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "source_index", "u", "v", "x", "y", "z"])
        for i, (source_index, uv, point) in enumerate(zip(result.source_indices, result.uv, result.points)):
            writer.writerow([
                i,
                int(source_index),
                f"{uv[0]:.10g}",
                f"{uv[1]:.10g}",
                f"{point[0]:.10g}",
                f"{point[1]:.10g}",
                f"{point[2]:.10g}",
            ])


def export_uv_ply(path: str | Path, result: UMAPResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uv_min = result.uv.min(axis=0)
    uv_span = np.maximum(np.ptp(result.uv, axis=0), EPS)
    colors_2d = np.clip((result.uv - uv_min) / uv_span, 0.0, 1.0)
    colors = np.column_stack([
        (255.0 * colors_2d[:, 0]).astype(np.uint8),
        (255.0 * colors_2d[:, 1]).astype(np.uint8),
        np.full(len(result.uv), 180, dtype=np.uint8),
    ])
    cloud = trimesh.points.PointCloud(result.points, colors=colors)
    cloud.export(str(path))


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius < EPS:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _mesh_face_sample(mesh: trimesh.Trimesh, max_faces: int = 6000) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) <= max_faces:
        return faces
    idx = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
    return faces[idx]


def visualize_result(
    source: PointSource,
    result: UMAPResult,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
    full_graph: bool = False,
) -> None:
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"UMAP UV - {source.name}", fontsize=13)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    if source.mesh is not None and len(source.mesh.faces) > 0:
        faces = _mesh_face_sample(source.mesh)
        vertices = np.asarray(source.mesh.vertices, dtype=float)
        collection = Poly3DCollection(
            vertices[faces],
            facecolor="#9fb7c9",
            edgecolor="#324756",
            linewidths=0.08,
            alpha=0.28,
        )
        ax3d.add_collection3d(collection)

    color = result.uv[:, 0]
    ax3d.scatter(result.points[:, 0], result.points[:, 1], result.points[:, 2], c=color, s=8, cmap="viridis", depthshade=False)
    ax3d.set_title("3D STL points colored by U")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    _set_axes_equal_3d(ax3d, source.points if source.mesh is None else np.asarray(source.mesh.vertices, dtype=float))

    axuv = fig.add_subplot(1, 2, 2)
    if len(result.graph_edges) > 0:
        edges = result.graph_edges if full_graph else result.graph_edges[: min(len(result.graph_edges), 12000)]
        segments = result.uv[edges]
        axuv.add_collection(LineCollection(segments, colors="#c9ced6", linewidths=0.25, alpha=0.55))
    scatter = axuv.scatter(result.uv[:, 0], result.uv[:, 1], c=result.uv[:, 1], s=9, cmap="plasma")
    axuv.set_aspect("equal", adjustable="datalim")
    axuv.set_title("2D UV coordinate system by UMAP")
    axuv.set_xlabel("u")
    axuv.set_ylabel("v")
    axuv.grid(True, color="#d0d7de", linewidth=0.4)
    fig.colorbar(scatter, ax=axuv, fraction=0.046, pad=0.04)

    info = [
        f"method: {result.method}",
        f"points: {len(result.points)}",
        f"graph edges: {len(result.graph_edges)}",
    ]
    fig.text(0.01, 0.01, "\n".join(info), fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(prefix.with_name(prefix.name + "_umap_uv.png"), dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(
    model_path: str | Path,
    n_neighbors: int,
    max_points: int,
    min_dist: float,
    spread: float,
    n_epochs: int,
    learning_rate: float,
    negative_sample_rate: int,
    random_state: int,
    out_csv: str | Path,
    out_ply: str | Path = "",
    fig_prefix: str | Path = "",
    no_show: bool = False,
    full_graph: bool = False,
    force_builtin: bool = False,
) -> UMAPResult:
    source = load_mesh_points(model_path)
    original_count = len(source.points)
    source = deterministic_sample(source, max_points=max_points)
    print(f"Loaded {original_count} STL mesh vertices from {model_path}")
    if len(source.points) != original_count:
        print(f"Using {len(source.points)} sampled points for UMAP. Increase --max-points for denser UV.")
    print(f"Computing UMAP UV with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    result = compute_umap_uv(
        source,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        random_state=random_state,
        force_builtin=force_builtin,
    )
    result.original_count = original_count
    result.sampled_count = len(source.points)
    export_uv_csv(out_csv, result)
    print(f"Saved UV CSV: {out_csv}")
    if out_ply:
        export_uv_ply(out_ply, result)
        print(f"Saved colored point cloud: {out_ply}")
    if not no_show or fig_prefix:
        visualize_result(source, result, output_prefix=fig_prefix or None, show=not no_show, full_graph=full_graph)
        if fig_prefix:
            prefix = Path(fig_prefix)
            print(f"Saved UV figure: {prefix.with_name(prefix.name + '_umap_uv.png')}")
    print(f"Method: {result.method}")
    print(f"Embedded points: {len(result.points)}")
    print(f"Graph edges: {len(result.graph_edges)}")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map STL mesh vertices to a 2D UV coordinate system using UMAP.")
    parser.add_argument("model", nargs="?", default="", help="Input STL/OBJ/PLY path. If omitted, a file dialog opens.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP nearest-neighbor count.")
    parser.add_argument("--max-points", type=int, default=1500, help="Maximum STL vertices sampled for UMAP.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist parameter.")
    parser.add_argument("--spread", type=float, default=1.0, help="UMAP spread parameter.")
    parser.add_argument("--epochs", type=int, default=150, help="Optimization epochs for the built-in UMAP fallback.")
    parser.add_argument("--learning-rate", type=float, default=1.0, help="Initial learning rate for the built-in UMAP fallback.")
    parser.add_argument("--negative-sample-rate", type=int, default=3, help="Negative samples per positive edge.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--force-builtin", action="store_true", help="Skip umap-learn and use the built-in fallback.")
    parser.add_argument("--out", default="umap_uv_points.csv", help="Output CSV path with u,v,x,y,z.")
    parser.add_argument("--out-ply", default="", help="Optional PLY point cloud colored by UV.")
    parser.add_argument("--fig-prefix", default="", help="Optional prefix for saving the PNG visualization.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    parser.add_argument("--full-graph", action="store_true", help="Draw all graph edges in the UV plot.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    model_path = Path(args.model) if args.model else select_model_file_dialog()
    if model_path is None:
        raise SystemExit("No model selected.")
    run_pipeline(
        model_path=model_path,
        n_neighbors=args.n_neighbors,
        max_points=args.max_points,
        min_dist=args.min_dist,
        spread=args.spread,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        negative_sample_rate=args.negative_sample_rate,
        random_state=args.random_state,
        out_csv=args.out,
        out_ply=args.out_ply,
        fig_prefix=args.fig_prefix,
        no_show=args.no_show,
        full_graph=args.full_graph,
        force_builtin=args.force_builtin,
    )


if __name__ == "__main__":
    main()
