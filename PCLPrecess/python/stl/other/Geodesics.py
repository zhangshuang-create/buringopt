#!/usr/bin/env python3
"""
Geodesics in Heat on STL meshes.

This is a compact implementation of the heat method:
1. Build the cotangent Laplacian and lumped mass matrix.
2. Diffuse a heat impulse from a source vertex for a short time.
3. Convert the heat gradient to a unit vector field.
4. Recover geodesic distance by solving a Poisson equation.

Run directly:
    py Geodesics.py

The script opens a file dialog for selecting an STL/OBJ/PLY mesh and visualizes
the geodesic distance field on the surface.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve


EPS = 1.0e-12


@dataclass
class HeatGeodesicResult:
    distances: np.ndarray
    heat: np.ndarray
    source_index: int
    time_step: float
    mean_edge_length: float
    surface_samples: Optional[np.ndarray] = None
    surface_centroid: Optional[np.ndarray] = None
    source_surface_point: Optional[np.ndarray] = None
    point_cloud_distances: Optional[np.ndarray] = None
    point_cloud_source_index: Optional[int] = None


@dataclass
class SurfaceCenterResult:
    points: np.ndarray
    face_indices: np.ndarray
    centroid: np.ndarray
    source_surface_point: np.ndarray
    source_vertex_index: int
    point_density: float
    graph: Optional[coo_matrix] = None
    source_point_index: int = 0
    point_distances: Optional[np.ndarray] = None


@dataclass
class MDSGeodesicResult:
    uv: np.ndarray
    sample_indices: np.ndarray
    distance_matrix: np.ndarray
    source_sample_index: int
    source_distances: np.ndarray
    eigenvalues: np.ndarray
    dense_uv: Optional[np.ndarray] = None
    dense_source_distances: Optional[np.ndarray] = None
    surface_point_xyz: Optional[np.ndarray] = None
    surface_point_uv: Optional[np.ndarray] = None
    surface_point_source_distances: Optional[np.ndarray] = None


@dataclass
class TopologyUVResult:
    uv: np.ndarray
    boundary_loops: list[np.ndarray]
    corner_indices: np.ndarray
    seam_vertices: np.ndarray
    mode: str


def select_mesh_file_dialog() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("tkinter is unavailable, cannot open file dialog.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select STL mesh for Geodesics in Heat",
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


def load_mesh(path: str | Path) -> trimesh.Trimesh:
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
        raise ValueError(f"Unsupported mesh type: {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The selected mesh has no vertices or faces.")

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
    return mesh


def subdivide_mesh(mesh: trimesh.Trimesh, steps: int) -> trimesh.Trimesh:
    refined = mesh.copy()
    for _ in range(max(0, int(steps))):
        vertices, faces = trimesh.remesh.subdivide(
            np.asarray(refined.vertices, dtype=float),
            np.asarray(refined.faces, dtype=np.int64),
        )
        refined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        refined.merge_vertices()
        refined.remove_unreferenced_vertices()
    return refined


def triangle_area(points: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def cotangent(a: np.ndarray, b: np.ndarray) -> float:
    cross_norm = float(np.linalg.norm(np.cross(a, b)))
    if cross_norm < EPS:
        return 0.0
    return float(np.dot(a, b) / cross_norm)


def build_cotangent_laplacian_and_mass(mesh: trimesh.Trimesh) -> Tuple[coo_matrix, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = len(vertices)

    edge_weights: dict[Tuple[int, int], float] = {}
    mass = np.zeros(vertex_count, dtype=float)

    def add_weight(i: int, j: int, value: float):
        if i > j:
            i, j = j, i
        edge_weights[(i, j)] = edge_weights.get((i, j), 0.0) + value

    for tri in faces:
        i, j, k = [int(x) for x in tri]
        pi, pj, pk = vertices[[i, j, k]]
        area = triangle_area(np.asarray([pi, pj, pk]))
        if area < EPS:
            continue
        mass[[i, j, k]] += area / 3.0

        cot_i = cotangent(pj - pi, pk - pi)
        cot_j = cotangent(pk - pj, pi - pj)
        cot_k = cotangent(pi - pk, pj - pk)

        add_weight(j, k, 0.5 * cot_i)
        add_weight(k, i, 0.5 * cot_j)
        add_weight(i, j, 0.5 * cot_k)

    rows = []
    cols = []
    vals = []
    diagonal = np.zeros(vertex_count, dtype=float)
    for (i, j), weight in edge_weights.items():
        if not np.isfinite(weight):
            continue
        diagonal[i] += weight
        diagonal[j] += weight
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([-weight, -weight])

    rows.extend(range(vertex_count))
    cols.extend(range(vertex_count))
    vals.extend(diagonal)
    stiffness = coo_matrix((vals, (rows, cols)), shape=(vertex_count, vertex_count))
    mass[mass < EPS] = EPS
    return stiffness, mass


def mean_edge_length(mesh: trimesh.Trimesh) -> float:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(edges) == 0:
        return 1.0
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    finite = lengths[np.isfinite(lengths) & (lengths > EPS)]
    return float(np.mean(finite)) if len(finite) else 1.0


def face_gradients(mesh: trimesh.Trimesh, scalar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    gradients = np.zeros((len(faces), 3), dtype=float)
    areas = np.zeros(len(faces), dtype=float)
    basis_grads = np.zeros((len(faces), 3, 3), dtype=float)

    for face_index, tri in enumerate(faces):
        p0, p1, p2 = vertices[tri]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < EPS:
            continue
        n = normal / norm
        area = 0.5 * norm
        areas[face_index] = area

        grad0 = np.cross(n, p2 - p1) / (2.0 * area)
        grad1 = np.cross(n, p0 - p2) / (2.0 * area)
        grad2 = np.cross(n, p1 - p0) / (2.0 * area)
        grads = np.asarray([grad0, grad1, grad2], dtype=float)
        basis_grads[face_index] = grads
        gradients[face_index] = scalar[tri] @ grads
    return gradients, areas, basis_grads


def integrated_divergence(mesh: trimesh.Trimesh, vector_field: np.ndarray, areas: np.ndarray, basis_grads: np.ndarray) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    divergence = np.zeros(len(mesh.vertices), dtype=float)
    for face_index, tri in enumerate(faces):
        if areas[face_index] < EPS:
            continue
        x = vector_field[face_index]
        for local, vertex in enumerate(tri):
            divergence[int(vertex)] += -areas[face_index] * float(np.dot(x, basis_grads[face_index, local]))
    return divergence


def solve_with_anchor(matrix: coo_matrix, rhs: np.ndarray, anchor_index: int, anchor_value: float = 0.0) -> np.ndarray:
    n = matrix.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[int(anchor_index)] = False
    reduced = matrix.tocsr()[mask][:, mask]
    reduced_rhs = np.asarray(rhs, dtype=float)[mask].copy()
    result = np.zeros(n, dtype=float)
    result[int(anchor_index)] = anchor_value
    solution = spsolve(reduced, reduced_rhs)
    result[mask] = solution
    return result


def compute_heat_geodesics(mesh: trimesh.Trimesh, source_index: int = 0, time_scale: float = 1.0) -> HeatGeodesicResult:
    vertex_count = len(mesh.vertices)
    source_index = int(np.clip(source_index, 0, vertex_count - 1))

    stiffness, mass = build_cotangent_laplacian_and_mass(mesh)
    h = mean_edge_length(mesh)
    t = float(time_scale) * h * h

    mass_matrix = diags(mass)
    heat_rhs = np.zeros(vertex_count, dtype=float)
    heat_rhs[source_index] = 1.0
    heat = spsolve((mass_matrix + t * stiffness).tocsr(), heat_rhs)

    grad_u, areas, basis_grads = face_gradients(mesh, heat)
    norms = np.linalg.norm(grad_u, axis=1)
    vector_field = np.zeros_like(grad_u)
    valid = norms > EPS
    vector_field[valid] = -grad_u[valid] / norms[valid, None]

    divergence = integrated_divergence(mesh, vector_field, areas, basis_grads)
    phi = solve_with_anchor(stiffness, divergence, source_index, anchor_value=0.0)
    distances = phi - phi[source_index]
    if float(np.mean(distances)) < 0.0:
        distances = -distances
    distances -= distances[source_index]
    distances[distances < 0.0] = 0.0

    return HeatGeodesicResult(
        distances=distances,
        heat=heat,
        source_index=source_index,
        time_step=t,
        mean_edge_length=h,
    )


def build_edge_length_graph(mesh: trimesh.Trimesh) -> coo_matrix:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("Mesh has no edges for geodesic distance graph.")
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([lengths, lengths])
    return coo_matrix((vals, (rows, cols)), shape=(len(vertices), len(vertices)))


def extract_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Return ordered boundary vertex loops; reject branched boundaries."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(directed, axis=1)
    unique_edges, counts = np.unique(undirected, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    invalid = [v for v, neighbors in adjacency.items() if len(neighbors) != 2]
    if invalid:
        raise ValueError(
            "Boundary is open or non-manifold: each boundary vertex must have degree 2 "
            f"(found {len(invalid)} invalid vertices)."
        )

    unused = {tuple(map(int, edge)) for edge in boundary_edges}
    loops: list[np.ndarray] = []
    while unused:
        start, current = min(unused)
        previous = start
        loop = [start]
        unused.discard(tuple(sorted((start, current))))
        while current != start:
            loop.append(current)
            next_vertex = next(v for v in adjacency[current] if v != previous)
            edge = tuple(sorted((current, next_vertex)))
            if edge not in unused and next_vertex != start:
                raise ValueError("Boundary has ambiguous or self-intersecting topology.")
            unused.discard(edge)
            previous, current = current, next_vertex
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def loop_arclength(vertices: np.ndarray, loop: np.ndarray) -> Tuple[np.ndarray, float]:
    points = vertices[loop]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))), float(np.sum(lengths))


def detect_boundary_corners(
    vertices: np.ndarray,
    loop: np.ndarray,
    angle_threshold_degrees: float = 25.0,
    scale_fraction: float = 0.02,
    suppression_fraction: float = 0.06,
) -> np.ndarray:
    count = len(loop)
    if count < 3:
        return np.empty(0, dtype=np.int64)
    step = max(1, min(count // 4, int(round(count * scale_fraction))))
    points = vertices[loop]
    incoming = points - np.roll(points, step, axis=0)
    outgoing = np.roll(points, -step, axis=0) - points
    incoming /= np.maximum(np.linalg.norm(incoming, axis=1, keepdims=True), EPS)
    outgoing /= np.maximum(np.linalg.norm(outgoing, axis=1, keepdims=True), EPS)
    angles = np.arccos(np.clip(np.einsum("ij,ij->i", incoming, outgoing), -1.0, 1.0))
    candidates = np.flatnonzero(angles >= np.deg2rad(angle_threshold_degrees))
    candidates = candidates[np.argsort(angles[candidates])[::-1]]
    radius = max(0, int(round(count * suppression_fraction)))
    selected: list[int] = []
    for candidate in candidates:
        distances = [min(abs(int(candidate) - i), count - abs(int(candidate) - i)) for i in selected]
        if not distances or min(distances) > radius:
            selected.append(int(candidate))
    return np.asarray(sorted(selected), dtype=np.int64)


def multisource_graph_distance(graph, sources: Sequence[int]) -> np.ndarray:
    distances, _ = multisource_graph_labels(graph, sources)
    return distances


def multisource_graph_labels(graph, sources: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    csr = graph.tocsr()
    distances = np.full(csr.shape[0], np.inf, dtype=float)
    labels = np.full(csr.shape[0], -1, dtype=np.int64)
    queue: list[tuple[float, int]] = []
    for label, source in enumerate(np.asarray(sources, dtype=np.int64)):
        distances[int(source)] = 0.0
        labels[int(source)] = label
        heapq.heappush(queue, (0.0, int(source)))
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance > distances[vertex]:
            continue
        for offset in range(csr.indptr[vertex], csr.indptr[vertex + 1]):
            neighbor = int(csr.indices[offset])
            candidate = distance + float(csr.data[offset])
            if candidate + EPS < distances[neighbor]:
                distances[neighbor] = candidate
                labels[neighbor] = labels[vertex]
                heapq.heappush(queue, (candidate, neighbor))
    return distances, labels


def shortest_path_vertices(graph, source: int, target: int) -> np.ndarray:
    distances, predecessors = dijkstra(
        graph.tocsr(), directed=False, indices=int(source), return_predecessors=True
    )
    if not np.isfinite(distances[int(target)]):
        raise ValueError("Cannot construct a seam across a disconnected mesh.")
    path = [int(target)]
    while path[-1] != int(source):
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("Shortest-path predecessor chain is incomplete.")
        path.append(predecessor)
    return np.asarray(path[::-1], dtype=np.int64)


def compute_topology_guided_uv(
    mesh: trimesh.Trimesh,
    corner_angle_degrees: float = 25.0,
) -> TopologyUVResult:
    """Build geodesic coordinates for annulus-like and disk-like open meshes."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    loops = extract_boundary_loops(mesh)
    if not loops:
        raise ValueError("The mesh has no boundary; a closed surface needs an explicit cut seed.")
    if len(loops) > 2:
        raise ValueError(
            f"Found {len(loops)} boundary loops; automatic UV supports one disk boundary "
            "or two annulus boundaries."
        )
    graph = build_edge_length_graph(mesh).tocsr()

    if len(loops) == 2:
        lengths = [loop_arclength(vertices, loop)[1] for loop in loops]
        outer_index = int(np.argmax(lengths))
        outer, inner = loops[outer_index], loops[1 - outer_index]
        arc, perimeter = loop_arclength(vertices, outer)
        seam_start = int(outer[0])
        from_start = dijkstra(graph, directed=False, indices=seam_start)
        seam_end = int(inner[np.argmin(from_start[inner])])
        seam = shortest_path_vertices(graph, seam_start, seam_end)

        # Wave collision opposite arc=0 is the periodic discontinuity of U.
        d_outer, nearest_outer = multisource_graph_labels(graph, outer)
        u = arc[nearest_outer]
        # V remains a physical geodesic length; scaling it by circumference would
        # incorrectly stretch a short cylinder.
        uv = np.column_stack((u, d_outer))
        return TopologyUVResult(uv, loops, np.empty(0, dtype=np.int64), seam, "annulus")

    loop = loops[0]
    corners_local = detect_boundary_corners(vertices, loop, angle_threshold_degrees=corner_angle_degrees)
    if len(corners_local) < 2:
        raise ValueError(
            f"Only {len(corners_local)} significant corners found; at least two are needed "
            "to define the longest boundary baseline."
        )
    arc, perimeter = loop_arclength(vertices, loop)
    segment_lengths = [
        (arc[corners_local[(i + 1) % len(corners_local)]] - arc[begin]) % perimeter
        for i, begin in enumerate(corners_local)
    ]
    longest = int(np.argmax(segment_lengths))
    begin = int(corners_local[longest])
    end = int(corners_local[(longest + 1) % len(corners_local)])
    baseline = loop[begin:end + 1] if begin < end else np.concatenate((loop[begin:], loop[:end + 1]))
    endpoint_distances = dijkstra(
        graph, directed=False, indices=[int(baseline[0]), int(baseline[-1])]
    )
    baseline_length = max(float(segment_lengths[longest]), EPS)
    u = 0.5 * (endpoint_distances[0] - endpoint_distances[1] + baseline_length)
    v = multisource_graph_distance(graph, baseline)
    uv = np.column_stack((u, v))
    return TopologyUVResult(uv, loops, loop[corners_local], np.empty(0, dtype=np.int64), "disk")


def export_topology_uv_csv(path: str | Path, mesh: trimesh.Trimesh, result: TopologyUVResult) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    boundary_ids = np.full(len(vertices), -1, dtype=np.int64)
    for loop_index, loop in enumerate(result.boundary_loops):
        boundary_ids[loop] = loop_index
    corners = set(map(int, result.corner_indices))
    seam = set(map(int, result.seam_vertices))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["vertex_index", "x", "y", "z", "u", "v", "boundary_loop", "is_corner", "is_seam"])
        for index, (point, uv) in enumerate(zip(vertices, result.uv)):
            writer.writerow([index, *point, *uv, int(boundary_ids[index]), int(index in corners), int(index in seam)])


def deterministic_sample_indices(points: np.ndarray, max_points: int, include_index: int) -> np.ndarray:
    point_count = len(points)
    include_index = int(np.clip(include_index, 0, point_count - 1))
    if max_points <= 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)

    max_points = max(2, int(max_points))
    selected = np.empty(max_points, dtype=np.int64)
    selected[0] = include_index
    min_dist2 = np.sum((points - points[include_index]) ** 2, axis=1)
    for i in range(1, max_points):
        selected[i] = int(np.argmax(min_dist2))
        dist2 = np.sum((points - points[selected[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.unique(selected)


def classical_mds(distance_matrix: np.ndarray, dimensions: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(distance_matrix, dtype=float)
    d2 = distances * distances
    row_mean = d2.mean(axis=1, keepdims=True)
    col_mean = d2.mean(axis=0, keepdims=True)
    total_mean = float(d2.mean())
    gram = -0.5 * (d2 - row_mean - col_mean + total_mean)

    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    values = values[:dimensions]
    vectors = vectors[:, :dimensions]
    uv = vectors * np.sqrt(np.maximum(values, 0.0))
    return uv, values


def distance_field_radial_uv(uv: np.ndarray, source_sample_index: int, source_distances: np.ndarray) -> np.ndarray:
    """Use MDS only for angular order, and use source geodesic distance as radius."""
    centered = np.asarray(uv, dtype=float).copy()
    centered -= centered[int(source_sample_index)]

    norms = np.linalg.norm(centered, axis=1)
    directions = np.zeros_like(centered)
    valid = norms > EPS
    directions[valid] = centered[valid] / norms[valid, None]

    if np.any(~valid):
        fallback_angles = np.linspace(0.0, 2.0 * np.pi, len(centered), endpoint=False)
        directions[~valid, 0] = np.cos(fallback_angles[~valid])
        directions[~valid, 1] = np.sin(fallback_angles[~valid])
    directions[int(source_sample_index)] = 0.0

    radii = np.asarray(source_distances, dtype=float).copy()
    radii -= float(np.min(radii))
    radius_scale = max(float(np.max(radii)), EPS)
    radii /= radius_scale
    return directions * radii[:, None]


def radial_uv_from_directions(directions: np.ndarray, source_distances: np.ndarray, source_index: int) -> np.ndarray:
    directions = np.asarray(directions, dtype=float).copy()
    norms = np.linalg.norm(directions, axis=1)
    valid = norms > EPS
    directions[valid] /= norms[valid, None]
    directions[~valid] = 0.0

    radii = np.asarray(source_distances, dtype=float).copy()
    radii -= float(np.min(radii))
    radius_scale = max(float(np.max(radii)), EPS)
    radii /= radius_scale

    uv = directions * radii[:, None]
    uv[int(source_index)] = 0.0
    return uv


def extend_radial_uv_to_all_vertices(
    vertices: np.ndarray,
    sample_indices: np.ndarray,
    sample_uv: np.ndarray,
    source_index: int,
    source_distances: np.ndarray,
    neighbors: int = 12,
) -> np.ndarray:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_points = np.asarray(vertices, dtype=float)[sample_indices]
    sample_uv = np.asarray(sample_uv, dtype=float)

    sample_norms = np.linalg.norm(sample_uv, axis=1)
    sample_dirs = np.zeros_like(sample_uv)
    valid = sample_norms > EPS
    sample_dirs[valid] = sample_uv[valid] / sample_norms[valid, None]

    if np.any(~valid):
        fallback_angles = np.linspace(0.0, 2.0 * np.pi, len(sample_uv), endpoint=False)
        sample_dirs[~valid, 0] = np.cos(fallback_angles[~valid])
        sample_dirs[~valid, 1] = np.sin(fallback_angles[~valid])

    k = int(np.clip(neighbors, 1, len(sample_indices)))
    tree = cKDTree(sample_points)
    try:
        nearest_distances, nearest = tree.query(vertices, k=k, workers=-1)
    except TypeError:
        nearest_distances, nearest = tree.query(vertices, k=k)
    if k == 1:
        nearest_distances = nearest_distances[:, None]
        nearest = nearest[:, None]

    weights = 1.0 / np.maximum(nearest_distances, EPS) ** 2
    complex_dirs = sample_dirs[:, 0] + 1j * sample_dirs[:, 1]
    blended = np.sum(weights * complex_dirs[nearest], axis=1) / np.maximum(np.sum(weights, axis=1), EPS)

    dense_dirs = np.zeros((len(vertices), 2), dtype=float)
    dense_norms = np.abs(blended)
    ok = dense_norms > EPS
    dense_dirs[ok, 0] = np.real(blended[ok]) / dense_norms[ok]
    dense_dirs[ok, 1] = np.imag(blended[ok]) / dense_norms[ok]

    fallback = ~ok
    if np.any(fallback):
        dense_dirs[fallback] = sample_dirs[nearest[fallback, 0]]
    dense_dirs[sample_indices] = sample_dirs
    dense_dirs[int(source_index)] = 0.0
    return radial_uv_from_directions(dense_dirs, source_distances, source_index)


def van_der_corput(index: int, base: int = 2) -> float:
    value = 0.0
    denom = 1.0
    index = int(index)
    while index > 0:
        index, remainder = divmod(index, base)
        denom *= base
        value += remainder / denom
    return value


def uniform_triangle_barycentric_samples(samples_per_face: int, offset: int = 0) -> np.ndarray:
    count = max(0, int(samples_per_face))
    if count <= 0:
        return np.zeros((0, 3), dtype=float)

    bary = np.zeros((count, 3), dtype=float)
    for i in range(count):
        u = (i + 0.5) / count
        v = van_der_corput(i + 1 + int(offset), base=2)
        sqrt_u = np.sqrt(u)
        bary[i, 0] = 1.0 - sqrt_u
        bary[i, 1] = sqrt_u * (1.0 - v)
        bary[i, 2] = sqrt_u * v
    return bary


def allocate_area_uniform_sample_counts(face_areas: np.ndarray, average_points_per_face: int, point_density: float) -> Tuple[np.ndarray, float]:
    areas = np.asarray(face_areas, dtype=float)
    counts = np.zeros(len(areas), dtype=np.int64)
    valid = np.isfinite(areas) & (areas > EPS)
    total_area = float(np.sum(areas[valid]))
    if total_area <= EPS:
        return counts, 0.0

    if point_density > 0.0:
        density = float(point_density)
        target_total = int(round(total_area * density))
    else:
        target_total = int(round(max(0, int(average_points_per_face)) * len(areas)))
        density = target_total / total_area if target_total > 0 else 0.0

    if target_total <= 0 or density <= 0.0:
        return counts, 0.0

    expected = np.zeros(len(areas), dtype=float)
    expected[valid] = areas[valid] * density
    counts = np.floor(expected).astype(np.int64)
    remainder = max(0, target_total - int(np.sum(counts)))
    if remainder > 0:
        fractional = expected - counts
        order = np.argsort(fractional)[::-1]
        counts[order[:remainder]] += 1
    return counts, density


def sample_uniform_points_on_faces(
    mesh: trimesh.Trimesh,
    dense_uv: np.ndarray,
    source_distances: np.ndarray,
    average_points_per_face: int,
    point_density: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_xyz = vertices[faces]
    face_uv = np.asarray(dense_uv, dtype=float)[faces]
    face_distances = np.asarray(source_distances, dtype=float)[faces]
    face_areas = 0.5 * np.linalg.norm(np.cross(face_xyz[:, 1] - face_xyz[:, 0], face_xyz[:, 2] - face_xyz[:, 0]), axis=1)
    counts, _density = allocate_area_uniform_sample_counts(face_areas, average_points_per_face, point_density)
    total_count = int(np.sum(counts))
    if total_count <= 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros(0, dtype=float),
        )

    xyz = np.empty((total_count, 3), dtype=float)
    uv = np.empty((total_count, 2), dtype=float)
    distances = np.empty(total_count, dtype=float)
    cursor = 0
    offset = 0
    for face_index, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        bary = uniform_triangle_barycentric_samples(count, offset=offset)
        next_cursor = cursor + count
        xyz[cursor:next_cursor] = bary @ face_xyz[face_index]
        uv[cursor:next_cursor] = bary @ face_uv[face_index]
        distances[cursor:next_cursor] = bary @ face_distances[face_index]
        cursor = next_cursor
        offset += count
    return xyz, uv, distances


def build_point_cloud_geodesic_graph(points: np.ndarray, neighbors: int = 12):
    """在等密度曲面点云上建立对称局部近邻图。"""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two surface points are required.")
    tree = cKDTree(points)
    maximum_neighbors = min(64, len(points) - 1)
    neighbor_count = min(max(2, int(neighbors)), maximum_neighbors)
    while True:
        distances, indices = tree.query(points, k=neighbor_count + 1, workers=-1)
        rows = np.repeat(np.arange(len(points), dtype=np.int64), neighbor_count)
        cols = indices[:, 1:].reshape(-1).astype(np.int64)
        values = distances[:, 1:].reshape(-1)
        valid = np.isfinite(values) & (values > EPS) & (rows != cols)
        rows, cols, values = rows[valid], cols[valid], values[valid]
        graph = coo_matrix(
            (np.concatenate((values, values)),
             (np.concatenate((rows, cols)), np.concatenate((cols, rows)))),
            shape=(len(points), len(points)),
        ).tocsr()
        component_count = connected_components(graph, directed=False, return_labels=False)
        if component_count == 1:
            return graph
        if neighbor_count >= maximum_neighbors:
            raise ValueError(f"Point-cloud neighbor graph has {component_count} disconnected components.")
        neighbor_count = min(maximum_neighbors, neighbor_count + 4)


def compute_area_uniform_surface_center(
    mesh: trimesh.Trimesh,
    average_points_per_face: int = 8,
    point_density: float = 0.0,
) -> SurfaceCenterResult:
    """按单位面积等密度采样，并以最小平均测地距离求曲面内蕴质心。"""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_xyz = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(face_xyz[:, 1] - face_xyz[:, 0], face_xyz[:, 2] - face_xyz[:, 0]),
        axis=1,
    )
    counts, density = allocate_area_uniform_sample_counts(
        face_areas,
        average_points_per_face=average_points_per_face,
        point_density=point_density,
    )
    total_count = int(np.sum(counts))
    if total_count <= 0:
        raise ValueError("Surface point-cloud density produced no samples.")

    points = np.empty((total_count, 3), dtype=float)
    point_faces = np.empty(total_count, dtype=np.int64)
    cursor = 0
    offset = 0
    for face_index, count_value in enumerate(counts):
        count = int(count_value)
        if count <= 0:
            continue
        barycentric = uniform_triangle_barycentric_samples(count, offset=offset)
        next_cursor = cursor + count
        points[cursor:next_cursor] = barycentric @ face_xyz[face_index]
        point_faces[cursor:next_cursor] = face_index
        cursor = next_cursor
        offset += count

    # 普通三维均值仅用于报告；带孔或弯曲曲面中它可能位于曲面之外。
    centroid = np.mean(points, axis=0)

    # Geodesics 和中心搜索都直接在面积均匀点云的局部近邻图上进行。
    point_graph = build_point_cloud_geodesic_graph(points, neighbors=12)
    euclidean_nearest = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    candidate_indices = deterministic_sample_indices(
        points,
        max_points=min(512, len(points)),
        include_index=euclidean_nearest,
    )
    best_score = float("inf")
    source_point_index = int(candidate_indices[0])
    batch_size = 16
    for begin in range(0, len(candidate_indices), batch_size):
        candidates = candidate_indices[begin:begin + batch_size]
        candidate_distances = dijkstra(
            point_graph,
            directed=False,
            indices=candidates,
            unweighted=False,
        )
        if not np.isfinite(candidate_distances).all():
            raise ValueError("The point cloud is disconnected; a unique geodesic center cannot be computed.")
        scores = np.mean(candidate_distances, axis=1)
        local_best = int(np.argmin(scores))
        if float(scores[local_best]) < best_score:
            best_score = float(scores[local_best])
            source_point_index = int(candidates[local_best])

    source_surface_point = points[source_point_index].copy()
    point_distances = np.asarray(
        dijkstra(point_graph, directed=False, indices=source_point_index, unweighted=False),
        dtype=float,
    )
    # Heat-on-mesh/MDS compatibility: select the closest vertex of the source point's face.
    source_face_index = int(point_faces[source_point_index])
    source_face_vertices = faces[source_face_index]
    source_vertex_index = int(source_face_vertices[np.argmin(
        np.sum((vertices[source_face_vertices] - source_surface_point) ** 2, axis=1)
    )])
    return SurfaceCenterResult(
        points=points,
        face_indices=point_faces,
        centroid=centroid,
        source_surface_point=source_surface_point,
        source_vertex_index=source_vertex_index,
        point_density=float(density),
        graph=point_graph,
        source_point_index=source_point_index,
        point_distances=point_distances,
    )


def compute_mds_from_geodesic_matrix(
    mesh: trimesh.Trimesh,
    source_index: int,
    max_points: int = 800,
    full_source_distances: Optional[np.ndarray] = None,
    dense_neighbors: int = 12,
    uv_points_per_face: int = 0,
    uv_point_density: float = 0.0,
) -> MDSGeodesicResult:
    vertices = np.asarray(mesh.vertices, dtype=float)
    sample_indices = deterministic_sample_indices(vertices, max_points=max_points, include_index=source_index)
    source_hits = np.where(sample_indices == int(source_index))[0]
    if len(source_hits) == 0:
        sample_indices = np.insert(sample_indices, 0, int(source_index))
        source_sample_index = 0
    else:
        source_sample_index = int(source_hits[0])

    graph = build_edge_length_graph(mesh).tocsr()
    distances_to_all = dijkstra(graph, directed=False, indices=sample_indices, unweighted=False)
    distance_matrix = distances_to_all[:, sample_indices]
    if not np.isfinite(distance_matrix).all():
        raise ValueError("The mesh edge graph is disconnected; cannot build a complete geodesic distance matrix for MDS.")

    source_distances = distance_matrix[source_sample_index]
    uv_raw, eigenvalues = classical_mds(distance_matrix, dimensions=2)
    uv = distance_field_radial_uv(uv_raw, source_sample_index, source_distances)
    dense_source_distances = (
        np.asarray(full_source_distances, dtype=float)
        if full_source_distances is not None
        else distances_to_all[source_sample_index]
    )
    dense_uv = extend_radial_uv_to_all_vertices(
        vertices,
        sample_indices,
        uv,
        source_index,
        dense_source_distances,
        neighbors=dense_neighbors,
    )
    surface_point_xyz = None
    surface_point_uv = None
    surface_point_source_distances = None
    if uv_points_per_face > 0 or uv_point_density > 0.0:
        surface_point_xyz, surface_point_uv, surface_point_source_distances = sample_uniform_points_on_faces(
            mesh,
            dense_uv,
            dense_source_distances,
            average_points_per_face=uv_points_per_face,
            point_density=uv_point_density,
        )
    return MDSGeodesicResult(
        uv=uv,
        sample_indices=sample_indices,
        distance_matrix=distance_matrix,
        source_sample_index=source_sample_index,
        source_distances=source_distances,
        eigenvalues=eigenvalues,
        dense_uv=dense_uv,
        dense_source_distances=dense_source_distances,
        surface_point_xyz=surface_point_xyz,
        surface_point_uv=surface_point_uv,
        surface_point_source_distances=surface_point_source_distances,
    )


def export_mds_uv_csv(path: str | Path, mesh: trimesh.Trimesh, mds: MDSGeodesicResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if (
            mds.surface_point_xyz is not None
            and mds.surface_point_uv is not None
            and mds.surface_point_source_distances is not None
            and len(mds.surface_point_uv) > 0
        ):
            writer.writerow(["point_index", "u", "v", "x", "y", "z", "source_geodesic_distance"])
            for point_i, point in enumerate(mds.surface_point_xyz):
                uv = mds.surface_point_uv[point_i]
                writer.writerow([
                    point_i,
                    f"{uv[0]:.10g}",
                    f"{uv[1]:.10g}",
                    f"{point[0]:.10g}",
                    f"{point[1]:.10g}",
                    f"{point[2]:.10g}",
                    f"{mds.surface_point_source_distances[point_i]:.10g}",
                ])
            return

        if mds.dense_uv is not None and mds.dense_source_distances is not None and len(mds.dense_uv) == len(vertices):
            sample_lookup = {int(vertex_i) for vertex_i in mds.sample_indices}
            writer.writerow(["vertex_index", "u", "v", "x", "y", "z", "source_geodesic_distance", "is_mds_sample"])
            for vertex_i, point in enumerate(vertices):
                uv = mds.dense_uv[vertex_i]
                writer.writerow([
                    vertex_i,
                    f"{uv[0]:.10g}",
                    f"{uv[1]:.10g}",
                    f"{point[0]:.10g}",
                    f"{point[1]:.10g}",
                    f"{point[2]:.10g}",
                    f"{mds.dense_source_distances[vertex_i]:.10g}",
                    int(vertex_i in sample_lookup),
                ])
            return

        writer.writerow(["sample_index", "vertex_index", "u", "v", "x", "y", "z", "source_geodesic_distance"])
        for sample_i, vertex_i in enumerate(mds.sample_indices):
            point = vertices[int(vertex_i)]
            uv = mds.uv[sample_i]
            writer.writerow([
                sample_i,
                int(vertex_i),
                f"{uv[0]:.10g}",
                f"{uv[1]:.10g}",
                f"{point[0]:.10g}",
                f"{point[1]:.10g}",
                f"{point[2]:.10g}",
                f"{mds.source_distances[sample_i]:.10g}",
            ])


def export_distances_csv(path: str | Path, mesh: trimesh.Trimesh, result: HeatGeodesicResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_index", "x", "y", "z", "geodesic_distance"])
        for i, (point, distance) in enumerate(zip(vertices, result.distances)):
            writer.writerow([i, f"{point[0]:.10g}", f"{point[1]:.10g}", f"{point[2]:.10g}", f"{distance:.10g}"])


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius < EPS:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def visualize_geodesics(
    mesh: trimesh.Trimesh,
    result: HeatGeodesicResult,
    mesh_path: Path,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    distances = result.distances
    point_cloud_mode = (
        result.surface_samples is not None
        and result.point_cloud_distances is not None
        and len(result.surface_samples) == len(result.point_cloud_distances)
    )
    displayed_distances = result.point_cloud_distances if point_cloud_mode else distances
    face_values = np.mean(distances[faces], axis=1)
    norm = colors.Normalize(vmin=float(np.min(displayed_distances)), vmax=float(np.max(displayed_distances)))
    cmap = cm.get_cmap("turbo")
    face_colors = cmap(norm(face_values))

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if point_cloud_mode:
        cloud = result.surface_samples
        cloud_distances = result.point_cloud_distances
        if len(cloud) > 120000:
            cloud_indices = np.linspace(0, len(cloud) - 1, 120000, dtype=np.int64)
            cloud = cloud[cloud_indices]
            cloud_distances = cloud_distances[cloud_indices]
        ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            c=cmap(norm(cloud_distances)), s=2.0, linewidths=0, depthshade=False,
        )
    else:
        collection = Poly3DCollection(
            vertices[faces],
            facecolors=face_colors,
            edgecolor="#24292f",
            linewidths=0.05,
            alpha=0.96,
        )
        ax.add_collection3d(collection)

    source = (
        result.surface_samples[int(result.point_cloud_source_index)]
        if point_cloud_mode and result.point_cloud_source_index is not None
        else vertices[result.source_index]
    )
    if (not point_cloud_mode) and result.surface_samples is not None and len(result.surface_samples):
        display_count = min(5000, len(result.surface_samples))
        display_indices = np.linspace(0, len(result.surface_samples) - 1, display_count, dtype=np.int64)
        cloud = result.surface_samples[display_indices]
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c="#202124", s=1, alpha=0.16, depthshade=False, label="uniform surface cloud")
    if result.surface_centroid is not None:
        centroid = result.surface_centroid
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c="#00a86b", marker="x", s=90, depthshade=False, label="intrinsic surface center")
    if result.source_surface_point is not None and (
        result.surface_centroid is None
        or np.linalg.norm(result.source_surface_point - result.surface_centroid) > EPS
    ):
        center_point = result.source_surface_point
        ax.scatter([center_point[0]], [center_point[1]], [center_point[2]], c="#ff9800", s=55, depthshade=False, label="nearest surface center")
    ax.scatter([source[0]], [source[1]], [source[2]], c="#ff2d2d", s=60, depthshade=False, label="source")

    ax.set_title(f"Geodesics in Heat - {mesh_path.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal_3d(ax, vertices)

    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(displayed_distances)
    cbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("geodesic distance")
    ax.legend(loc="upper right")

    info = (
        f"source vertex: {result.source_index}\n"
        f"mean edge length h: {result.mean_edge_length:.6g}\n"
        f"heat time t: {result.time_step:.6g}\n"
        f"max distance: {float(np.max(distances)):.6g}"
    )
    fig.text(0.01, 0.01, info, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(prefix.with_name(prefix.name + "_heat_geodesics.png"), dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def wave_face_colors(distances: np.ndarray, faces: np.ndarray, phase: float, band_width: float) -> np.ndarray:
    face_distances = np.mean(distances[faces], axis=1)
    delta = face_distances - phase
    wave = np.exp(-(delta * delta) / (2.0 * band_width * band_width))

    previous = face_distances <= phase
    base = np.zeros((len(face_distances), 4), dtype=float)
    base[:, 0] = np.where(previous, 0.10, 0.78)
    base[:, 1] = np.where(previous, 0.36, 0.82)
    base[:, 2] = np.where(previous, 0.72, 0.88)
    base[:, 3] = 0.94

    crest = np.column_stack([
        np.ones(len(wave)),
        0.88 * np.ones(len(wave)),
        0.10 * np.ones(len(wave)),
        np.ones(len(wave)),
    ])
    return base * (1.0 - wave[:, None]) + crest * wave[:, None]


def wave_point_colors(distances: np.ndarray, phase: float, band_width: float) -> np.ndarray:
    delta = np.asarray(distances, dtype=float) - float(phase)
    wave = np.exp(-(delta * delta) / (2.0 * band_width * band_width))
    previous = distances <= phase
    base = np.zeros((len(distances), 4), dtype=float)
    base[:, 0] = np.where(previous, 0.12, 0.78)
    base[:, 1] = np.where(previous, 0.35, 0.82)
    base[:, 2] = np.where(previous, 0.70, 0.88)
    base[:, 3] = np.where(previous, 0.92, 0.22)
    crest = np.column_stack([
        np.ones(len(wave)),
        0.88 * np.ones(len(wave)),
        0.10 * np.ones(len(wave)),
        np.ones(len(wave)),
    ])
    return base * (1.0 - wave[:, None]) + crest * wave[:, None]


def sampled_faces_for_render(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces
    indices = np.linspace(0, len(faces) - 1, int(max_faces), dtype=np.int64)
    return faces[indices]


def animate_geodesic_wave(
    mesh: trimesh.Trimesh,
    result: HeatGeodesicResult,
    mesh_path: Path,
    mds: Optional[MDSGeodesicResult] = None,
    frame_count: int = 140,
    interval_ms: int = 45,
    render_max_faces: int = 12000,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    faces = sampled_faces_for_render(faces_all, render_max_faces)
    distances = result.distances
    left_distances = (
        result.point_cloud_distances
        if result.point_cloud_distances is not None
        else distances
    )
    left_max_distance = max(float(np.max(left_distances)), EPS)
    uv_distances = mds.surface_point_source_distances if mds is not None and mds.surface_point_source_distances is not None else None
    if uv_distances is None and mds is not None:
        uv_distances = mds.dense_source_distances
    if uv_distances is None and mds is not None:
        uv_distances = mds.source_distances
    right_max_distance = max(float(np.max(uv_distances)), EPS) if uv_distances is not None else left_max_distance
    max_distance = max(left_max_distance, right_max_distance, EPS)
    band_width = max(max_distance / 32.0, result.mean_edge_length * 0.35, EPS)

    fig = plt.figure(figsize=(15, 7.5))
    if mds is None:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        axuv = None
    else:
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        axuv = fig.add_subplot(1, 2, 2)
    point_cloud_mode = (
        result.surface_samples is not None
        and result.point_cloud_distances is not None
        and len(result.surface_samples) == len(result.point_cloud_distances)
    )
    if point_cloud_mode:
        cloud = result.surface_samples
        cloud_distances = result.point_cloud_distances
        if len(cloud) > 120000:
            cloud_indices = np.linspace(0, len(cloud) - 1, 120000, dtype=np.int64)
            cloud = cloud[cloud_indices]
            cloud_distances = cloud_distances[cloud_indices]
        collection = ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            s=2.0,
            c=wave_point_colors(cloud_distances, 0.0, band_width),
            linewidths=0,
            depthshade=False,
        )
    else:
        collection = Poly3DCollection(
            vertices[faces],
            facecolors=wave_face_colors(distances, faces, 0.0, band_width),
            edgecolor="#1f2933",
            linewidths=0.04,
            alpha=0.98,
        )
        ax.add_collection3d(collection)

    source = (
        result.surface_samples[int(result.point_cloud_source_index)]
        if point_cloud_mode and result.point_cloud_source_index is not None
        else vertices[result.source_index]
    )
    if (not point_cloud_mode) and result.surface_samples is not None and len(result.surface_samples):
        display_count = min(5000, len(result.surface_samples))
        display_indices = np.linspace(0, len(result.surface_samples) - 1, display_count, dtype=np.int64)
        cloud = result.surface_samples[display_indices]
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c="#111111", s=0.8, alpha=0.12, depthshade=False, label="uniform cloud")
    if result.surface_centroid is not None:
        centroid = result.surface_centroid
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c="#00a86b", marker="x", s=80, depthshade=False, label="surface center")
    if result.source_surface_point is not None and (
        result.surface_centroid is None
        or np.linalg.norm(result.source_surface_point - result.surface_centroid) > EPS
    ):
        center_point = result.source_surface_point
        ax.scatter([center_point[0]], [center_point[1]], [center_point[2]], c="#ff9800", s=50, depthshade=False, label="surface center")
    ax.scatter([source[0]], [source[1]], [source[2]], c="#ff2d2d", s=70, depthshade=False, label="source")
    ax.set_title(f"Geodesic wave propagation - {mesh_path.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_axes_equal_3d(ax, vertices)
    ax.legend(loc="upper right")

    uv_artist = None
    if mds is not None and axuv is not None:
        if (
            mds.surface_point_uv is not None
            and mds.surface_point_source_distances is not None
            and len(mds.surface_point_uv) > 0
        ):
            uv_artist = axuv.scatter(
                mds.surface_point_uv[:, 0],
                mds.surface_point_uv[:, 1],
                s=3,
                c=wave_point_colors(mds.surface_point_source_distances, 0.0, band_width),
                linewidths=0,
            )
            source_uv = mds.dense_uv[result.source_index] if mds.dense_uv is not None else mds.uv[mds.source_sample_index]
            margin = 0.05
            uv_min = np.min(mds.surface_point_uv, axis=0)
            uv_max = np.max(mds.surface_point_uv, axis=0)
            uv_span = np.maximum(uv_max - uv_min, EPS)
            axuv.set_xlim(uv_min[0] - margin * uv_span[0], uv_max[0] + margin * uv_span[0])
            axuv.set_ylim(uv_min[1] - margin * uv_span[1], uv_max[1] + margin * uv_span[1])
        elif mds.dense_uv is not None and mds.dense_source_distances is not None and len(mds.dense_uv) == len(vertices):
            uv_faces = faces_all
            uv_artist = PolyCollection(
                mds.dense_uv[uv_faces],
                facecolors=wave_face_colors(mds.dense_source_distances, uv_faces, 0.0, band_width),
                edgecolors="none",
                closed=True,
                alpha=0.96,
            )
            axuv.add_collection(uv_artist)
            source_uv = mds.dense_uv[result.source_index]
            margin = 0.05
            uv_min = np.min(mds.dense_uv, axis=0)
            uv_max = np.max(mds.dense_uv, axis=0)
            uv_span = np.maximum(uv_max - uv_min, EPS)
            axuv.set_xlim(uv_min[0] - margin * uv_span[0], uv_max[0] + margin * uv_span[0])
            axuv.set_ylim(uv_min[1] - margin * uv_span[1], uv_max[1] + margin * uv_span[1])
        else:
            uv_artist = axuv.scatter(
                mds.uv[:, 0],
                mds.uv[:, 1],
                s=12,
                c=wave_point_colors(mds.source_distances, 0.0, band_width),
            )
            source_uv = mds.uv[mds.source_sample_index]
        axuv.scatter([source_uv[0]], [source_uv[1]], c="#ff2d2d", s=42, zorder=4, label="source")
        axuv.set_title("Radial UV: radius = geodesic distance")
        axuv.set_xlabel("u")
        axuv.set_ylabel("v")
        axuv.set_aspect("equal", adjustable="datalim")
        axuv.grid(True, color="#d0d7de", linewidth=0.4)
        axuv.legend(loc="upper right")

    text = fig.text(0.01, 0.01, "", fontsize=9, va="bottom")

    def update(frame: int):
        phase = (frame / max(1, frame_count - 1)) * (max_distance + 2.0 * band_width)
        if point_cloud_mode:
            collection.set_facecolor(wave_point_colors(cloud_distances, phase, band_width))
        else:
            collection.set_facecolor(wave_face_colors(distances, faces, phase, band_width))
        if uv_artist is not None and mds is not None:
            if (
                mds.surface_point_source_distances is not None
                and mds.surface_point_uv is not None
                and len(mds.surface_point_uv) > 0
                and not isinstance(uv_artist, PolyCollection)
            ):
                uv_artist.set_facecolor(wave_point_colors(mds.surface_point_source_distances, phase, band_width))
            elif isinstance(uv_artist, PolyCollection) and mds.dense_source_distances is not None:
                uv_artist.set_facecolor(wave_face_colors(mds.dense_source_distances, faces_all, phase, band_width))
            else:
                uv_artist.set_facecolor(wave_point_colors(mds.source_distances, phase, band_width))
        text.set_text(
            f"source vertex: {result.source_index}\n"
            f"wavefront distance: {min(phase, max_distance):.6g} / {max_distance:.6g}\n"
            f"band width: {band_width:.6g}"
            + ("" if mds is None else f"\nMDS samples: {len(mds.sample_indices)}")
            + (
                ""
                if mds is None or mds.surface_point_uv is None
                else f"\nUV point cloud: {len(mds.surface_point_uv)}"
            )
        )
        if uv_artist is not None:
            return collection, uv_artist, text
        return collection, text

    animation = FuncAnimation(fig, update, frames=int(frame_count), interval=int(interval_ms), blit=False, repeat=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        try:
            animation.save(prefix.with_name(prefix.name + "_heat_wave.gif"), writer="pillow", fps=max(1, int(1000 / interval_ms)))
        except Exception as exc:
            print(f"Could not save GIF animation: {exc}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def pyvista_wave_scalars(distances: np.ndarray, phase: float, band_width: float) -> np.ndarray:
    delta = np.asarray(distances, dtype=float) - phase
    wave = np.exp(-(delta * delta) / (2.0 * band_width * band_width))
    previous = distances <= phase
    return np.where(previous, 0.35, 0.05) + 0.65 * wave


def animate_geodesic_wave_pyvista(
    mesh: trimesh.Trimesh,
    result: HeatGeodesicResult,
    mesh_path: Path,
    mds: Optional[MDSGeodesicResult] = None,
    frame_count: int = 140,
    interval_ms: int = 45,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
) -> bool:
    try:
        import pyvista as pv  # type: ignore
    except Exception:
        return False

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    distances = result.distances
    left_max_distance = max(float(np.max(distances)), EPS)
    uv_distances = mds.surface_point_source_distances if mds is not None and mds.surface_point_source_distances is not None else None
    if uv_distances is None and mds is not None:
        uv_distances = mds.dense_source_distances
    if uv_distances is None and mds is not None:
        uv_distances = mds.source_distances
    right_max_distance = max(float(np.max(uv_distances)), EPS) if uv_distances is not None else left_max_distance
    max_distance = max(left_max_distance, right_max_distance, EPS)
    band_width = max(max_distance / 32.0, result.mean_edge_length * 0.35, EPS)

    pv_faces = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).reshape(-1)
    pv_mesh = pv.PolyData(vertices, pv_faces)
    pv_mesh.cell_data["wave"] = pyvista_wave_scalars(np.mean(distances[faces], axis=1), 0.0, band_width)

    shape = (1, 2) if mds is not None else (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=(1500, 780))
    plotter.subplot(0, 0)
    plotter.add_text(f"Geodesic wave - {mesh_path.name}", font_size=10)
    plotter.add_mesh(pv_mesh, scalars="wave", cmap="turbo", clim=(0.0, 1.0), show_edges=False, smooth_shading=False)
    if result.surface_samples is not None and len(result.surface_samples):
        display_count = min(5000, len(result.surface_samples))
        display_indices = np.linspace(0, len(result.surface_samples) - 1, display_count, dtype=np.int64)
        plotter.add_points(result.surface_samples[display_indices], color="black", opacity=0.16, point_size=2)
    if result.surface_centroid is not None:
        plotter.add_points(np.asarray([result.surface_centroid]), color="green", point_size=18, render_points_as_spheres=True)
    if result.source_surface_point is not None and (
        result.surface_centroid is None
        or np.linalg.norm(result.source_surface_point - result.surface_centroid) > EPS
    ):
        plotter.add_points(np.asarray([result.source_surface_point]), color="orange", point_size=16, render_points_as_spheres=True)
    plotter.add_points(vertices[[result.source_index]], color="red", point_size=14, render_points_as_spheres=True)
    plotter.show_axes()

    uv_cloud = None
    uv_is_dense_surface = False
    uv_is_uniform_point_cloud = False
    if mds is not None:
        plotter.subplot(0, 1)
        plotter.add_text("Radial UV: radius = geodesic distance", font_size=10)
        if (
            mds.surface_point_uv is not None
            and mds.surface_point_source_distances is not None
            and len(mds.surface_point_uv) > 0
        ):
            uv_points = np.column_stack([mds.surface_point_uv, np.zeros(len(mds.surface_point_uv))])
            uv_cloud = pv.PolyData(uv_points)
            uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.surface_point_source_distances, 0.0, band_width)
            plotter.add_mesh(uv_cloud, scalars="wave", cmap="turbo", clim=(0.0, 1.0), point_size=4, render_points_as_spheres=True)
            source_uv = mds.dense_uv[result.source_index] if mds.dense_uv is not None else mds.uv[mds.source_sample_index]
            plotter.add_points(np.asarray([[source_uv[0], source_uv[1], 0.0]]), color="red", point_size=14, render_points_as_spheres=True)
            uv_is_uniform_point_cloud = True
        elif mds.dense_uv is not None and mds.dense_source_distances is not None and len(mds.dense_uv) == len(vertices):
            uv_points = np.column_stack([mds.dense_uv, np.zeros(len(mds.dense_uv))])
            uv_cloud = pv.PolyData(uv_points, pv_faces)
            uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.dense_source_distances, 0.0, band_width)
            plotter.add_mesh(uv_cloud, scalars="wave", cmap="turbo", clim=(0.0, 1.0), show_edges=False, smooth_shading=False)
            plotter.add_points(uv_points[[result.source_index]], color="red", point_size=14, render_points_as_spheres=True)
            uv_is_dense_surface = True
        else:
            uv_points = np.column_stack([mds.uv, np.zeros(len(mds.uv))])
            uv_cloud = pv.PolyData(uv_points)
            uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.source_distances, 0.0, band_width)
            plotter.add_mesh(uv_cloud, scalars="wave", cmap="turbo", clim=(0.0, 1.0), point_size=7, render_points_as_spheres=True)
            plotter.add_points(uv_points[[mds.source_sample_index]], color="red", point_size=14, render_points_as_spheres=True)
        plotter.view_xy()
        plotter.show_axes()

    face_distances = np.mean(distances[faces], axis=1)

    def update_frame(frame: int):
        phase = (frame / max(1, frame_count - 1)) * (max_distance + 2.0 * band_width)
        pv_mesh.cell_data["wave"] = pyvista_wave_scalars(face_distances, phase, band_width)
        if uv_cloud is not None and mds is not None:
            if uv_is_uniform_point_cloud and mds.surface_point_source_distances is not None:
                uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.surface_point_source_distances, phase, band_width)
            elif uv_is_dense_surface and mds.dense_source_distances is not None:
                uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.dense_source_distances, phase, band_width)
            else:
                uv_cloud.point_data["wave"] = pyvista_wave_scalars(mds.source_distances, phase, band_width)

    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        try:
            plotter.open_gif(str(prefix.with_name(prefix.name + "_heat_wave_gpu.gif")), fps=max(1, int(1000 / interval_ms)))
            for frame in range(int(frame_count)):
                update_frame(frame)
                plotter.write_frame()
            plotter.close()
            return True
        except Exception as exc:
            print(f"Could not save PyVista GIF: {exc}")

    if show:
        state = {"frame": 0}

        def timer_callback(*_args):
            update_frame(state["frame"])
            state["frame"] = (state["frame"] + 1) % int(frame_count)

        plotter.add_timer_event(max_steps=0, duration=int(interval_ms), callback=timer_callback)
        plotter.show()
    else:
        plotter.close()
    return True


def run_pipeline(
    mesh_path: str | Path,
    source_index: int,
    time_scale: float,
    out_csv: str | Path = "heat_geodesic_distances.csv",
    mds_out_csv: str | Path = "heat_mds_uv.csv",
    fig_prefix: str | Path = "",
    no_show: bool = False,
    animate: bool = True,
    frames: int = 140,
    interval_ms: int = 45,
    mds: bool = True,
    mds_max_points: int = 2000,
    dense_neighbors: int = 12,
    uv_points_per_face: int = 8,
    uv_point_density: float = 0.0,
    subdivide_steps: int = 0,
    render_backend: str = "auto",
    render_max_faces: int = 12000,
    topology_uv: bool = False,
    topology_uv_out: str | Path = "topology_uv.csv",
    corner_angle_degrees: float = 25.0,
    center_points_per_face: int = 8,
    center_point_density: float = 0.0,
) -> HeatGeodesicResult:
    mesh_path = Path(mesh_path)
    mesh = load_mesh(mesh_path)
    if subdivide_steps > 0:
        print(f"Subdividing mesh, steps={subdivide_steps}...")
        mesh = subdivide_mesh(mesh, subdivide_steps)
    print(f"Loaded mesh: {mesh_path}")
    print(f"Vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}")
    print("Sampling an area-uniform surface point cloud...")
    center_result = compute_area_uniform_surface_center(
        mesh,
        average_points_per_face=center_points_per_face,
        point_density=center_point_density,
    )
    if int(source_index) < 0:
        source_index = center_result.source_vertex_index
    print(
        f"Surface samples: {len(center_result.points)}, "
        f"density={center_result.point_density:.6g} points/unit^2"
    )
    print(f"Euclidean point-cloud centroid (may be off-surface): {center_result.centroid}")
    print(f"Intrinsic surface center (geodesic medoid): {center_result.source_surface_point}")
    print(f"Computing heat geodesics from center source vertex {source_index}...")
    result = compute_heat_geodesics(mesh, source_index=source_index, time_scale=time_scale)
    result.surface_samples = center_result.points
    result.surface_centroid = center_result.source_surface_point
    result.source_surface_point = center_result.source_surface_point
    result.point_cloud_distances = center_result.point_distances
    result.point_cloud_source_index = center_result.source_point_index
    export_distances_csv(out_csv, mesh, result)
    print(f"Saved distances CSV: {out_csv}")
    print(f"Mean edge length h: {result.mean_edge_length:.6g}")
    print(f"Heat time t: {result.time_step:.6g}")
    print(f"Max geodesic distance: {float(np.max(result.distances)):.6g}")

    if topology_uv:
        topology_result = compute_topology_guided_uv(
            mesh, corner_angle_degrees=corner_angle_degrees
        )
        export_topology_uv_csv(topology_uv_out, mesh, topology_result)
        print(
            f"Topology UV: mode={topology_result.mode}, "
            f"boundaries={len(topology_result.boundary_loops)}, "
            f"corners={len(topology_result.corner_indices)}, "
            f"seam_vertices={len(topology_result.seam_vertices)}"
        )
        print(f"Saved topology-guided UV CSV: {topology_uv_out}")

    mds_result: Optional[MDSGeodesicResult] = None
    if mds:
        print(f"Building geodesic distance matrix for MDS, max samples={mds_max_points}...")
        mds_result = compute_mds_from_geodesic_matrix(
            mesh,
            result.source_index,
            max_points=mds_max_points,
            full_source_distances=result.distances,
            dense_neighbors=dense_neighbors,
            uv_points_per_face=uv_points_per_face,
            uv_point_density=uv_point_density,
        )
        export_mds_uv_csv(mds_out_csv, mesh, mds_result)
        print(f"Saved MDS UV CSV: {mds_out_csv}")
        print(f"MDS samples: {len(mds_result.sample_indices)}")
        if mds_result.dense_uv is not None:
            print(f"Dense UV vertices: {len(mds_result.dense_uv)}")
        if mds_result.surface_point_uv is not None:
            total_area = max(float(mesh.area), EPS)
            actual_density = len(mds_result.surface_point_uv) / total_area
            print(f"Uniform UV point cloud: {len(mds_result.surface_point_uv)} points, density={actual_density:.6g} points/unit^2")
        print(f"MDS eigenvalues: {mds_result.eigenvalues[0]:.6g}, {mds_result.eigenvalues[1]:.6g}")

    if animate and (not no_show or fig_prefix):
        backend = render_backend.lower().strip()
        rendered = False
        if backend in {"auto", "pyvista", "gpu"} and result.point_cloud_distances is None:
            rendered = animate_geodesic_wave_pyvista(
                mesh,
                result,
                mesh_path,
                mds=mds_result,
                frame_count=frames,
                interval_ms=interval_ms,
                output_prefix=fig_prefix or None,
                show=not no_show,
            )
            if not rendered and backend in {"pyvista", "gpu"}:
                print("PyVista/VTK is not available; falling back to Matplotlib CPU rendering.")
        if not rendered:
            animate_geodesic_wave(
                mesh,
                result,
                mesh_path,
                mds=mds_result,
                frame_count=frames,
                interval_ms=interval_ms,
                render_max_faces=render_max_faces,
                output_prefix=fig_prefix or None,
                show=not no_show,
            )
        if fig_prefix:
            prefix = Path(fig_prefix)
            print(f"Saved animation if supported: {prefix.with_name(prefix.name + '_heat_wave.gif')}")
    elif not no_show or fig_prefix:
        visualize_geodesics(mesh, result, mesh_path, output_prefix=fig_prefix or None, show=not no_show)
        if fig_prefix:
            prefix = Path(fig_prefix)
            print(f"Saved figure: {prefix.with_name(prefix.name + '_heat_geodesics.png')}")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize geodesic distances using Geodesics in Heat.")
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY path. If omitted, a file dialog opens.")
    parser.add_argument(
        "--source-index",
        type=int,
        default=-1,
        help="Source vertex index. Default -1 automatically uses the area-uniform point-cloud center.",
    )
    parser.add_argument(
        "--center-points-per-face",
        type=int,
        default=8,
        help="Target average point count per face for the area-uniform center point cloud.",
    )
    parser.add_argument(
        "--center-point-density",
        type=float,
        default=0.0,
        help="Exact target point density per unit surface area; overrides --center-points-per-face when > 0.",
    )
    parser.add_argument("--time-scale", type=float, default=1.0, help="Heat time t = time_scale * mean_edge_length^2.")
    parser.add_argument("--out", default="heat_geodesic_distances.csv", help="Output CSV path.")
    parser.add_argument("--mds-out", default="heat_mds_uv.csv", help="Output CSV path for MDS UV coordinates.")
    parser.add_argument("--mds-max-points", type=int, default=2000, help="Maximum sampled vertices used for MDS distance matrix.")
    parser.add_argument("--dense-neighbors", type=int, default=12, help="Nearest MDS samples used to interpolate dense UV directions.")
    parser.add_argument(
        "--uv-points-per-face",
        "--uv-average-points-per-face",
        dest="uv_points_per_face",
        type=int,
        default=8,
        help="Target average UV points per face; actual per-face counts are allocated by face area. Use 0 to disable unless --uv-point-density is set.",
    )
    parser.add_argument("--uv-point-density", type=float, default=0.0, help="UV point density in points per mesh unit^2. Overrides the average-points target when > 0.")
    parser.add_argument("--subdivide", type=int, default=0, help="Loop-free triangle subdivision steps before computing geodesics. 1 makes roughly 4x faces.")
    parser.add_argument("--no-mds", action="store_true", help="Disable the right-side MDS UV animation.")
    parser.add_argument("--fig-prefix", default="", help="Optional prefix for saving PNG visualization.")
    parser.add_argument("--static", action="store_true", help="Show static distance field instead of animated wave.")
    parser.add_argument("--frames", type=int, default=140, help="Animation frame count.")
    parser.add_argument("--interval", type=int, default=45, help="Animation interval in milliseconds.")
    parser.add_argument("--render-backend", choices=["auto", "matplotlib", "pyvista", "gpu"], default="auto", help="Animation renderer. pyvista/gpu uses VTK OpenGL if installed.")
    parser.add_argument("--render-max-faces", type=int, default=12000, help="Max faces refreshed by Matplotlib animation. Use 0 for all faces.")
    parser.add_argument("--topology-uv", action="store_true", help="Build boundary-aware geodesic UV coordinates for disk/annulus meshes.")
    parser.add_argument("--topology-uv-out", default="topology_uv.csv", help="Output CSV for boundary-aware UV coordinates.")
    parser.add_argument("--corner-angle", type=float, default=25.0, help="Minimum multiscale boundary turning angle in degrees.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    mesh_path = Path(args.mesh) if args.mesh else select_mesh_file_dialog()
    if mesh_path is None:
        raise SystemExit("No mesh selected.")
    run_pipeline(
        mesh_path=mesh_path,
        source_index=args.source_index,
        time_scale=args.time_scale,
        out_csv=args.out,
        mds_out_csv=args.mds_out,
        fig_prefix=args.fig_prefix,
        no_show=args.no_show,
        animate=not args.static,
        frames=args.frames,
        interval_ms=args.interval,
        mds=not args.no_mds,
        mds_max_points=args.mds_max_points,
        dense_neighbors=args.dense_neighbors,
        uv_points_per_face=args.uv_points_per_face,
        uv_point_density=args.uv_point_density,
        subdivide_steps=args.subdivide,
        render_backend=args.render_backend,
        render_max_faces=args.render_max_faces,
        topology_uv=args.topology_uv,
        topology_uv_out=args.topology_uv_out,
        corner_angle_degrees=args.corner_angle,
        center_points_per_face=args.center_points_per_face,
        center_point_density=args.center_point_density,
    )


if __name__ == "__main__":
    main()
