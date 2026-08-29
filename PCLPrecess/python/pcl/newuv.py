#!/usr/bin/env python3
"""
Geodesic-neighborhood UV visualization by Isomap.

This script imports an STL/OBJ/PLY mesh or a PCD point cloud, builds a k-nearest
neighbor graph in 3D, approximates geodesic distances with graph shortest paths,
then applies Classical MDS to unfold the sampled surface into a 2D UV plane.

Dependencies:
    py -m pip install numpy scipy trimesh matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


EPS = 1.0e-12


@dataclass
class PointSource:
    points: np.ndarray
    mesh: Optional[trimesh.Trimesh]
    source_indices: np.ndarray
    name: str


@dataclass
class IsomapUVResult:
    points: np.ndarray
    uv: np.ndarray
    source_indices: np.ndarray
    eigenvalues: np.ndarray
    stress: float
    graph_edges: np.ndarray
    distance_matrix: np.ndarray
    component_count: int
    original_count: int


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
        title="Select model or point cloud for Isomap UV",
        filetypes=[
            ("Model and point cloud files", "*.stl *.obj *.ply *.pcd"),
            ("Mesh files", "*.stl *.obj *.ply"),
            ("Point cloud files", "*.pcd"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def load_mesh_points(path: Path) -> PointSource:
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


def _pcd_header_and_payload(path: Path) -> Tuple[Dict[str, str], bytes]:
    data = path.read_bytes()
    lines: List[str] = []
    offset = 0
    for raw_line in data.splitlines(keepends=True):
        try:
            line = raw_line.decode("ascii", errors="ignore").strip()
        except UnicodeDecodeError:
            line = ""
        lines.append(line)
        offset += len(raw_line)
        if line.upper().startswith("DATA"):
            break
    else:
        raise ValueError(f"PCD file {path} has no DATA line.")

    header: Dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            header[parts[0].upper()] = parts[1].strip()
    return header, data[offset:]


def _pcd_numpy_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    dtype_fields = []
    for field, size, type_code, count in zip(fields, sizes, types, counts):
        if type_code == "F":
            base = {4: "<f4", 8: "<f8"}.get(size)
        elif type_code == "I":
            base = {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"}.get(size)
        elif type_code == "U":
            base = {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}.get(size)
        else:
            base = None
        if base is None:
            raise ValueError(f"Unsupported PCD field type: field={field}, size={size}, type={type_code}.")
        dtype_fields.append((field, base, (count,)) if count > 1 else (field, base))
    return np.dtype(dtype_fields)


def load_pcd_points(path: Path) -> PointSource:
    header, payload = _pcd_header_and_payload(path)
    fields = header.get("FIELDS", "").split()
    if not {"x", "y", "z"}.issubset(set(fields)):
        raise ValueError("PCD must contain x y z fields.")

    sizes = [int(x) for x in header.get("SIZE", "").split()]
    types = header.get("TYPE", "").split()
    counts = [int(x) for x in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    point_count = int(header.get("POINTS", header.get("WIDTH", "0")))
    data_mode = header.get("DATA", "").lower()

    if len(sizes) != len(fields) or len(types) != len(fields) or len(counts) != len(fields):
        raise ValueError("Invalid PCD header: FIELDS, SIZE, TYPE, COUNT lengths do not match.")

    if data_mode == "ascii":
        text = payload.decode("utf-8", errors="ignore").strip()
        raw = np.loadtxt(text.splitlines(), dtype=float)
        raw = np.atleast_2d(raw)
        x_idx, y_idx, z_idx = fields.index("x"), fields.index("y"), fields.index("z")
        points = raw[:, [x_idx, y_idx, z_idx]]
    elif data_mode == "binary":
        dtype = _pcd_numpy_dtype(fields, sizes, types, counts)
        rows = np.frombuffer(payload, dtype=dtype, count=point_count)
        points = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(float)
    elif data_mode == "binary_compressed":
        raise ValueError("PCD DATA binary_compressed is not supported by this script.")
    else:
        raise ValueError(f"Unsupported PCD DATA mode: {data_mode!r}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError(f"No finite xyz points found in {path}.")
    return PointSource(points=points, mesh=None, source_indices=np.arange(len(points)), name=path.name)


def load_point_source(path: str | Path) -> PointSource:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pcd":
        return load_pcd_points(path)
    if suffix in {".stl", ".obj", ".ply", ".glb", ".gltf"}:
        return load_mesh_points(path)
    raise ValueError(f"Unsupported input type {suffix!r}. Use STL/OBJ/PLY/PCD.")


def deterministic_sample(source: PointSource, max_points: int) -> PointSource:
    points = source.points
    if max_points <= 0 or len(points) <= max_points:
        return source

    # Farthest-point sampling gives a better global surface preview than taking
    # every nth vertex, while still being deterministic for repeatable UV plots.
    selected = np.empty(max_points, dtype=np.int64)
    center = points.mean(axis=0)
    selected[0] = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    min_dist2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, max_points):
        selected[i] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    selected.sort()

    sampled_points = points[selected]
    sampled_mesh = None
    if source.mesh is not None and len(source.mesh.faces) > 0:
        # The mesh is still useful for the 3D background plot, even though UV is
        # computed only on sampled vertices.
        sampled_mesh = source.mesh
    return PointSource(
        points=sampled_points,
        mesh=sampled_mesh,
        source_indices=source.source_indices[selected],
        name=source.name,
    )


def build_knn_graph(points: np.ndarray, k: int) -> Tuple[coo_matrix, np.ndarray]:
    n = len(points)
    if n < 3:
        raise ValueError("At least three points are required.")
    k = min(max(1, int(k)), n - 1)

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)

    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_weights = np.concatenate([weights, weights])
    graph = coo_matrix((all_weights, (all_rows, all_cols)), shape=(n, n)).tocsr()
    graph.sum_duplicates()
    edge_rows, edge_cols = graph.nonzero()
    keep = edge_rows < edge_cols
    graph_edges = np.column_stack([edge_rows[keep], edge_cols[keep]])
    return graph, graph_edges


def largest_connected_component(points: np.ndarray, graph, source_indices: np.ndarray) -> Tuple[np.ndarray, object, np.ndarray, int]:
    component_count, labels = connected_components(graph, directed=False)
    if component_count <= 1:
        return points, graph, source_indices, component_count

    counts = np.bincount(labels)
    largest = int(np.argmax(counts))
    keep = labels == largest
    reduced_points = points[keep]
    reduced_graph = graph[keep][:, keep]
    reduced_indices = source_indices[keep]
    return reduced_points, reduced_graph, reduced_indices, component_count


def classical_mds(distance_matrix: np.ndarray, dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    d2 = distance_matrix * distance_matrix
    row_mean = d2.mean(axis=1, keepdims=True)
    col_mean = d2.mean(axis=0, keepdims=True)
    total_mean = float(d2.mean())
    gram = -0.5 * (d2 - row_mean - col_mean + total_mean)

    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = values > max(EPS, abs(float(values[0])) * 1.0e-12)
    if np.count_nonzero(positive) < dimensions:
        raise ValueError("The geodesic distance matrix has fewer than two positive MDS eigenvalues.")
    values = values[:dimensions]
    vectors = vectors[:, :dimensions]
    values = np.maximum(values, 0.0)
    uv = vectors * np.sqrt(values)
    return uv, values


def normalize_uv(uv: np.ndarray) -> np.ndarray:
    out = uv.astype(float, copy=True)
    out -= out.mean(axis=0)
    span = np.ptp(out, axis=0)
    scale = max(float(np.max(span)), EPS)
    out /= scale
    return out


def relative_stress(uv: np.ndarray, distance_matrix: np.ndarray) -> float:
    embedded = np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=2)
    numerator = np.sum((embedded - distance_matrix) ** 2)
    denominator = np.sum(distance_matrix ** 2)
    if denominator < EPS:
        return 0.0
    return float(math.sqrt(numerator / denominator))


def compute_isomap_uv(source: PointSource, k: int) -> IsomapUVResult:
    graph, edges = build_knn_graph(source.points, k)
    points, graph, source_indices, component_count = largest_connected_component(source.points, graph, source.source_indices)
    if len(points) != len(source.points):
        graph, edges = build_knn_graph(points, min(k, len(points) - 1))
    distances = dijkstra(graph, directed=False, unweighted=False)
    if not np.isfinite(distances).all():
        raise ValueError("kNN graph is still disconnected after component filtering. Increase --k.")

    uv_raw, values = classical_mds(distances, dimensions=2)
    uv = normalize_uv(uv_raw)
    stress = relative_stress(uv_raw, distances)
    return IsomapUVResult(
        points=points,
        uv=uv,
        source_indices=source_indices,
        eigenvalues=values,
        stress=stress,
        graph_edges=edges,
        distance_matrix=distances,
        component_count=component_count,
        original_count=len(source.points),
    )


def export_uv_csv(path: str | Path, result: IsomapUVResult) -> None:
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


def export_uv_ply(path: str | Path, result: IsomapUVResult) -> None:
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
    result: IsomapUVResult,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
    full_graph: bool = False,
) -> None:
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Isomap geodesic UV - {source.name}", fontsize=13)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    if source.mesh is not None and len(source.mesh.faces) > 0:
        faces = _mesh_face_sample(source.mesh)
        vertices = np.asarray(source.mesh.vertices, dtype=float)
        collection = Poly3DCollection(
            vertices[faces],
            facecolor="#9fb7c9",
            edgecolor="#324756",
            linewidths=0.08,
            alpha=0.32,
        )
        ax3d.add_collection3d(collection)

    color = result.uv[:, 0]
    ax3d.scatter(result.points[:, 0], result.points[:, 1], result.points[:, 2], c=color, s=8, cmap="viridis", depthshade=False)
    ax3d.set_title("3D model / point cloud colored by u")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    _set_axes_equal_3d(ax3d, source.points if source.mesh is None else np.asarray(source.mesh.vertices, dtype=float))

    axuv = fig.add_subplot(1, 2, 2)
    if len(result.graph_edges) > 0:
        edges = result.graph_edges if full_graph else result.graph_edges[: min(len(result.graph_edges), 12000)]
        segments = result.uv[edges]
        axuv.add_collection(LineCollection(segments, colors="#c9ced6", linewidths=0.25, alpha=0.65))
    scatter = axuv.scatter(result.uv[:, 0], result.uv[:, 1], c=result.uv[:, 1], s=9, cmap="plasma")
    axuv.set_aspect("equal", adjustable="datalim")
    axuv.set_title("2D UV unfolding by graph geodesic Isomap")
    axuv.set_xlabel("u")
    axuv.set_ylabel("v")
    axuv.grid(True, color="#d0d7de", linewidth=0.4)
    fig.colorbar(scatter, ax=axuv, fraction=0.046, pad=0.04)

    info = [
        f"original points: {result.original_count}",
        f"embedded points: {len(result.points)}",
        f"kNN connected components before filtering: {result.component_count}",
        f"MDS eigenvalues: {result.eigenvalues[0]:.6g}, {result.eigenvalues[1]:.6g}",
        f"relative distance stress: {result.stress:.6g}",
    ]
    fig.text(0.01, 0.01, "\n".join(info), fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(prefix.with_name(prefix.name + "_isomap_uv.png"), dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_pipeline(
    model_path: str | Path,
    k: int,
    max_points: int,
    out_csv: str | Path,
    out_ply: str | Path = "",
    fig_prefix: str | Path = "",
    no_show: bool = False,
    full_graph: bool = False,
) -> IsomapUVResult:
    source = load_point_source(model_path)
    original_count = len(source.points)
    source = deterministic_sample(source, max_points=max_points)
    print(f"Loaded {original_count} points from {model_path}")
    if len(source.points) != original_count:
        print(f"Using {len(source.points)} sampled points for Isomap. Increase --max-points for denser UV.")
    print(f"Building kNN graph with k={k}...")
    result = compute_isomap_uv(source, k=k)
    export_uv_csv(out_csv, result)
    print(f"Saved UV CSV: {out_csv}")
    if out_ply:
        export_uv_ply(out_ply, result)
        print(f"Saved colored point cloud: {out_ply}")
    if not no_show or fig_prefix:
        visualize_result(source, result, output_prefix=fig_prefix or None, show=not no_show, full_graph=full_graph)
        if fig_prefix:
            prefix = Path(fig_prefix)
            print(f"Saved UV figure: {prefix.with_name(prefix.name + '_isomap_uv.png')}")
    print(f"Embedded points: {len(result.points)}")
    print(f"Connected components before filtering: {result.component_count}")
    print(f"MDS eigenvalues: {result.eigenvalues[0]:.6g}, {result.eigenvalues[1]:.6g}")
    print(f"Relative distance stress: {result.stress:.6g}")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unfold a 3D model/point cloud to UV using graph-geodesic Isomap.")
    parser.add_argument("model", nargs="?", default="", help="Input STL/OBJ/PLY/PCD path. If omitted, a file dialog opens.")
    parser.add_argument("--k", type=int, default=12, help="Nearest-neighbor count for the geodesic graph.")
    parser.add_argument("--max-points", type=int, default=2500, help="Maximum points used by full Isomap MDS.")
    parser.add_argument("--out", default="isomap_uv_points.csv", help="Output CSV path with u,v,x,y,z.")
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
        k=args.k,
        max_points=args.max_points,
        out_csv=args.out,
        out_ply=args.out_ply,
        fig_prefix=args.fig_prefix,
        no_show=args.no_show,
        full_graph=args.full_graph,
    )


if __name__ == "__main__":
    main()
