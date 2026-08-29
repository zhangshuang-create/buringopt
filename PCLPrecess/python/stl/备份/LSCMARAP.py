#!/usr/bin/env python3
"""LSCM initialization followed by flip-aware ARAP UV optimization.

Run this file directly. It opens a file dialog for selecting an STL/OBJ/PLY
mesh, computes a least-squares conformal mapping (LSCM), and plots the 3D mesh
next to its 2D (u, v) parameter domain.

The implemented steps follow the paper's Section 2.2:
1. Query triangle vertex indices.
2. Build each triangle's local coordinate system.
3. Compute triangle area and the MT matrix in Eq. (5).
4. Detect boundary vertices and lock two boundary vertices.
5. Assemble Eq. (4) as a sparse least-squares system.
6. Solve for all vertex coordinates in the planar parameter domain.
7. Refine the map with local-global ARAP iterations.

For a loft mesh with exactly two open boundary loops, the PCA reference plane
and start reference curve from loft_uv_parameterization.py define a longitudinal
seam.  Seam vertices are duplicated before parameterization so the annulus has
a disk-like topology suitable for texture mapping.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
import csv
import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(RUNTIME_CACHE / "cupy"))
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    from cupyx.scipy.sparse.linalg import factorized as gpu_factorized
except Exception:
    cp = None
    gpu_csr_matrix = None
    gpu_factorized = None


EPS = 1.0e-12


@dataclass
class LSCMResult:
    uv: np.ndarray
    locked_vertices: Tuple[int, int]
    boundary_vertices: np.ndarray
    energy_rms: float
    arap_energy_rms: float = 0.0
    arap_iterations: int = 0
    flipped_faces: int = 0
    mean_edge_stretch: float = 0.0
    max_edge_stretch: float = 0.0
    seam_vertices: np.ndarray = None
    compute_backend: str = "CPU"
    warning: str = ""


@dataclass
class PathPlanningResult:
    uv_paths: List[np.ndarray]
    xyz_paths: List[np.ndarray]
    line_count: int
    path_spacing: float


@dataclass
class CutMeshResult:
    mesh: trimesh.Trimesh
    source_vertex_indices: np.ndarray
    seam_vertices: np.ndarray
    seam_points: np.ndarray
    boundary_count_before: int
    boundary_count_after: int


def _unit(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vector / norm


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


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
        title="Select STL mesh for LSCM UV mapping",
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
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}.")
        mesh = trimesh.util.concatenate(meshes)
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


def boundary_vertices_from_faces(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edge_count: dict[Tuple[int, int], int] = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            edge_count[(ia, ib)] = edge_count.get((ia, ib), 0) + 1

    boundary_edges = np.asarray([edge for edge, count in edge_count.items() if count == 1], dtype=np.int64)
    if len(boundary_edges) == 0:
        return np.empty(0, dtype=np.int64), boundary_edges.reshape(0, 2)
    return np.unique(boundary_edges.reshape(-1)), boundary_edges


def ordered_boundary_loops(faces: np.ndarray) -> List[np.ndarray]:
    _, boundary_edges = boundary_vertices_from_faces(faces)
    if len(boundary_edges) == 0:
        return []
    adjacency: dict[int, List[int]] = {}
    unused = set()
    for a, b in boundary_edges:
        ia, ib = int(a), int(b)
        adjacency.setdefault(ia, []).append(ib)
        adjacency.setdefault(ib, []).append(ia)
        unused.add((min(ia, ib), max(ia, ib)))

    loops: List[np.ndarray] = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        loop = [start, current]
        unused.remove(first_edge)
        previous = start
        while current != start:
            candidates = [v for v in adjacency[current] if v != previous]
            next_vertex = None
            for candidate in candidates:
                edge = (min(current, candidate), max(current, candidate))
                if edge in unused:
                    next_vertex = candidate
                    unused.remove(edge)
                    break
            if next_vertex is None:
                break
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _loft_reference_seam(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, object]:
    import loft_uv_parameterization as loft

    loops = loft.extract_boundary_loops(mesh)
    top, bottom = loft.classify_top_bottom_boundaries(loops)
    _, axes, _ = loft.compute_pca(top.points)
    pca_second = axes[:, 1]
    try:
        third_point = loft._upper_boundary_contact_point(top)
    except ValueError:
        third_point = loft._upper_boundary_pca_second_point(top, pca_second)
    plane = loft.build_reference_plane(top.center, bottom.center, third_point)
    curves = loft.intersect_mesh_with_plane(mesh, plane)
    curve = loft.choose_start_reference_curve(curves, pca_second)
    return np.asarray(curve, dtype=float), plane


def _nearest_loop_vertex_to_curve(
    vertices: np.ndarray, loop: np.ndarray, curve: np.ndarray
) -> int:
    distances, _ = cKDTree(curve).query(vertices[loop], k=1)
    return int(loop[int(np.argmin(distances))])


def _guided_mesh_path(
    mesh: trimesh.Trimesh,
    start: int,
    end: int,
    guide_curve: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = mesh_edges_from_faces(np.asarray(mesh.faces, dtype=np.int64))
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    guide_distance, _ = cKDTree(guide_curve).query(midpoints, k=1)
    edge_length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    scale = max(float(np.median(edge_length)), EPS)
    weights = edge_length * (1.0 + 8.0 * guide_distance / scale)
    graph = coo_matrix(
        (
            np.concatenate([weights, weights]),
            (
                np.concatenate([edges[:, 0], edges[:, 1]]),
                np.concatenate([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    distance, predecessors = dijkstra(
        graph, directed=False, indices=start, return_predecessors=True
    )
    if not np.isfinite(distance[end]):
        raise ValueError("Could not find a mesh-edge seam between the two boundaries")
    path = [int(end)]
    while path[-1] != start:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("The seam predecessor chain is incomplete")
        path.append(predecessor)
    return np.asarray(path[::-1], dtype=np.int64)


def _faces_to_duplicate_along_seam(
    faces: np.ndarray, seam: np.ndarray
) -> dict[int, set[int]]:
    """Choose one incident face fan at every seam vertex.

    Removing the two seam half-edges from an interior vertex splits its
    incident face fan into two parts.  The selected parts are propagated along
    the path so both ends of every seam edge duplicate the same adjacent face.
    This is purely topological and therefore remains valid when a guided path
    crosses or locally deviates from its reference plane.
    """
    seam = np.asarray(seam, dtype=np.int64)
    if len(seam) < 2 or len(np.unique(seam)) != len(seam):
        raise ValueError(
            "The guided seam must be a simple path with at least two vertices"
        )

    seam_edges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(seam[:-1], seam[1:])
    }
    incident: dict[int, list[int]] = {int(v): [] for v in seam}
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(faces):
        tri_vertices = [int(v) for v in tri]
        for vertex in tri_vertices:
            if vertex in incident:
                incident[vertex].append(face_index)
        for a, b in ((tri_vertices[0], tri_vertices[1]),
                     (tri_vertices[1], tri_vertices[2]),
                     (tri_vertices[2], tri_vertices[0])):
            edge = (min(a, b), max(a, b))
            if edge in seam_edges:
                edge_faces.setdefault(edge, []).append(face_index)

    components: dict[int, list[set[int]]] = {}
    for vertex in map(int, seam):
        local_faces = incident[vertex]
        adjacency = {face_index: set() for face_index in local_faces}
        non_seam_edge_faces: dict[tuple[int, int], list[int]] = {}
        for face_index in local_faces:
            tri = [int(v) for v in faces[face_index]]
            others = [v for v in tri if v != vertex]
            for other in others:
                edge = (min(vertex, other), max(vertex, other))
                if edge not in seam_edges:
                    non_seam_edge_faces.setdefault(edge, []).append(face_index)
        for shared_faces in non_seam_edge_faces.values():
            for first in shared_faces:
                adjacency[first].update(f for f in shared_faces if f != first)

        remaining = set(local_faces)
        vertex_components: list[set[int]] = []
        while remaining:
            stack = [remaining.pop()]
            component: set[int] = set()
            while stack:
                face_index = stack.pop()
                component.add(face_index)
                new_faces = adjacency[face_index] & remaining
                remaining.difference_update(new_faces)
                stack.extend(new_faces)
            vertex_components.append(component)
        if len(vertex_components) != 2:
            raise ValueError(
                "The guided seam is not locally cuttable at vertex "
                f"{vertex} (expected 2 incident face fans, found {len(vertex_components)})."
            )
        components[vertex] = vertex_components

    duplicate_faces: dict[int, set[int]] = {}
    for path_index, (a_raw, b_raw) in enumerate(zip(seam[:-1], seam[1:])):
        a, b = int(a_raw), int(b_raw)
        edge = (min(a, b), max(a, b))
        adjacent_faces = edge_faces.get(edge, [])
        if len(adjacent_faces) != 2:
            raise ValueError(
                f"Seam edge {edge} must have exactly two adjacent faces, "
                f"found {len(adjacent_faces)}."
            )
        if path_index == 0:
            duplicate_faces[a] = next(
                component for component in components[a]
                if adjacent_faces[0] in component
            )
        selected_face = next(
            face_index for face_index in adjacent_faces
            if face_index in duplicate_faces[a]
        )
        duplicate_faces[b] = next(
            component for component in components[b]
            if selected_face in component
        )
    return duplicate_faces


def cut_two_boundary_loft(mesh: trimesh.Trimesh) -> CutMeshResult:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    loops = ordered_boundary_loops(faces)
    if len(loops) != 2:
        raise ValueError(f"Loft seam cutting requires exactly two boundaries, found {len(loops)}")

    guide_curve, _ = _loft_reference_seam(mesh)
    endpoints = [_nearest_loop_vertex_to_curve(vertices, loop, guide_curve) for loop in loops]
    seam = _guided_mesh_path(mesh, endpoints[0], endpoints[1], guide_curve)
    duplicate_faces = _faces_to_duplicate_along_seam(faces, seam)
    duplicate_map = {vertex: len(vertices) + i for i, vertex in enumerate(seam)}
    cut_faces = faces.copy()
    for face_index, tri in enumerate(cut_faces):
        for local_index, vertex_raw in enumerate(tri):
            vertex = int(vertex_raw)
            if vertex in duplicate_faces and face_index in duplicate_faces[vertex]:
                tri[local_index] = duplicate_map[vertex]

    cut_vertices = np.vstack([vertices, vertices[seam]])
    source_indices = np.concatenate([np.arange(len(vertices), dtype=np.int64), seam])
    cut_mesh = trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces, process=False)
    cut_mesh.remove_unreferenced_vertices()

    # remove_unreferenced_vertices may renumber, so recover source indices by
    # nearest exact coordinate. Duplicate seam copies intentionally share source IDs.
    source_tree = cKDTree(vertices)
    _, recovered = source_tree.query(np.asarray(cut_mesh.vertices), k=1)
    source_indices = np.asarray(recovered, dtype=np.int64)
    seam_mask = np.isin(source_indices, seam)
    cut_seam_vertices = np.flatnonzero(seam_mask)
    after_count = len(ordered_boundary_loops(np.asarray(cut_mesh.faces, dtype=np.int64)))
    if after_count != 1:
        raise ValueError(
            "The guided loft seam did not produce a disk-like mesh "
            f"(expected 1 boundary after cutting, found {after_count})."
        )
    return CutMeshResult(
        mesh=cut_mesh,
        source_vertex_indices=source_indices,
        seam_vertices=cut_seam_vertices,
        seam_points=vertices[seam],
        boundary_count_before=2,
        boundary_count_after=after_count,
    )


def prepare_parameterization_mesh(mesh: trimesh.Trimesh) -> CutMeshResult:
    boundary_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
    if boundary_count == 2:
        return cut_two_boundary_loft(mesh)
    return CutMeshResult(
        mesh=mesh,
        source_vertex_indices=np.arange(len(mesh.vertices), dtype=np.int64),
        seam_vertices=np.empty(0, dtype=np.int64),
        seam_points=np.empty((0, 3), dtype=float),
        boundary_count_before=boundary_count,
        boundary_count_after=boundary_count,
    )


def mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = []
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            edges.append((ia, ib))
    return np.unique(np.asarray(edges, dtype=np.int64), axis=0)


def choose_locked_vertices(vertices: np.ndarray, boundary_vertices: np.ndarray) -> Tuple[int, int, str]:
    warning = ""
    candidates = np.asarray(boundary_vertices, dtype=np.int64)
    if len(candidates) < 2:
        raise ValueError("LSCM requires at least two boundary vertices PBOU, as in the paper.")

    seed = int(candidates[0])
    distances = np.linalg.norm(vertices[candidates] - vertices[seed], axis=1)
    first = int(candidates[int(np.argmax(distances))])
    distances = np.linalg.norm(vertices[candidates] - vertices[first], axis=1)
    second = int(candidates[int(np.argmax(distances))])
    if first == second:
        raise ValueError("Cannot find two distinct vertices to lock for LSCM.")
    return first, second, warning


def local_triangle_coordinates(points: np.ndarray) -> Tuple[np.ndarray, float]:
    p1, p2, p3 = points
    x_axis = _unit(p2 - p1, "triangle local X axis")
    normal = _unit(np.cross(x_axis, p3 - p1), "triangle normal")
    y_axis = _unit(np.cross(normal, x_axis), "triangle local Y axis")

    rel = points - p1
    xy = np.column_stack([rel @ x_axis, rel @ y_axis])
    area = 0.5 * abs(_cross2d(xy[1] - xy[0], xy[2] - xy[0]))
    if area < EPS:
        raise ValueError("Degenerate triangle with near-zero area.")
    return xy, area


def triangle_gradient_matrix(local_xy: np.ndarray, area: float) -> np.ndarray:
    x1, y1 = local_xy[0]
    x2, y2 = local_xy[1]
    x3, y3 = local_xy[2]
    return np.asarray(
        [
            [y2 - y3, y3 - y1, y1 - y2],
            [x3 - x2, x1 - x3, x2 - x1],
        ],
        dtype=float,
    ) / (2.0 * area)


def assemble_lscm_system(vertices: np.ndarray, faces: np.ndarray) -> Tuple[coo_matrix, np.ndarray]:
    row_indices = []
    col_indices = []
    values = []
    row = 0
    skipped = 0
    vertex_count = len(vertices)

    for tri in faces:
        tri = np.asarray(tri, dtype=np.int64)
        try:
            local_xy, area = local_triangle_coordinates(vertices[tri])
        except ValueError:
            skipped += 1
            continue
        mt = triangle_gradient_matrix(local_xy, area)
        d_dx = mt[0]
        d_dy = mt[1]
        weight = math.sqrt(area)

        # Eq. (4): minimize (u_x - v_y)^2 + (v_x + u_y)^2 on each triangle.
        for local, vertex in enumerate(tri):
            row_indices.extend([row, row, row + 1, row + 1])
            col_indices.extend([int(vertex), vertex_count + int(vertex), int(vertex), vertex_count + int(vertex)])
            values.extend([
                weight * d_dx[local],
                -weight * d_dy[local],
                weight * d_dy[local],
                weight * d_dx[local],
            ])
        row += 2

    if row == 0:
        raise ValueError("No non-degenerate triangles are available for LSCM.")
    if skipped:
        warnings.warn(f"Skipped {skipped} degenerate triangles.")
    matrix = coo_matrix((values, (row_indices, col_indices)), shape=(row, 2 * vertex_count))
    return matrix, np.zeros(row, dtype=float)


def solve_lscm(mesh: trimesh.Trimesh) -> LSCMResult:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_vertices, _ = boundary_vertices_from_faces(faces)
    lock0, lock1, warning = choose_locked_vertices(vertices, boundary_vertices)

    matrix, rhs = assemble_lscm_system(vertices, faces)
    vertex_count = len(vertices)
    fixed_values = {
        lock0: 0.0,
        vertex_count + lock0: 0.0,
        lock1: 1.0,
        vertex_count + lock1: 0.0,
    }

    fixed_cols = np.asarray(sorted(fixed_values), dtype=np.int64)
    fixed_x = np.asarray([fixed_values[int(col)] for col in fixed_cols], dtype=float)
    all_cols = np.arange(2 * vertex_count, dtype=np.int64)
    free_mask = np.ones(2 * vertex_count, dtype=bool)
    free_mask[fixed_cols] = False
    free_cols = all_cols[free_mask]

    csr = matrix.tocsr()
    free_matrix = csr[:, free_cols]
    fixed_matrix = csr[:, fixed_cols]
    free_rhs = rhs - fixed_matrix @ fixed_x
    solution = lsqr(free_matrix, free_rhs, atol=1.0e-10, btol=1.0e-10, iter_lim=max(2000, 2 * vertex_count))[0]

    full = np.zeros(2 * vertex_count, dtype=float)
    full[fixed_cols] = fixed_x
    full[free_cols] = solution
    uv = np.column_stack([full[:vertex_count], full[vertex_count:]])
    uv = normalize_uv(uv, lock0, lock1)

    residual = csr @ full
    energy_rms = float(np.sqrt(np.mean(residual * residual))) if len(residual) else 0.0
    return LSCMResult(
        uv=uv,
        locked_vertices=(lock0, lock1),
        boundary_vertices=boundary_vertices,
        energy_rms=energy_rms,
        warning=warning,
    )


def normalize_uv(uv: np.ndarray, lock0: int, lock1: int) -> np.ndarray:
    out = np.asarray(uv, dtype=float).copy()
    out -= out[lock0]
    direction = out[lock1] - out[lock0]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    c, s = math.cos(-angle), math.sin(-angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    out = out @ rotation.T
    scale = float(np.linalg.norm(out[lock1] - out[lock0]))
    if scale > EPS:
        out /= scale
    return out


def _triangle_local_data(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.empty((len(faces), 3, 2), dtype=float)
    gradients = np.empty((len(faces), 2, 3), dtype=float)
    areas = np.empty(len(faces), dtype=float)
    for face_index, tri in enumerate(faces):
        xy, area = local_triangle_coordinates(vertices[tri])
        local[face_index] = xy
        gradients[face_index] = triangle_gradient_matrix(xy, area)
        areas[face_index] = area
    return local, gradients, areas


def _uv_signed_double_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = uv[faces]
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    return edge0[:, 0] * edge1[:, 1] - edge0[:, 1] * edge1[:, 0]


def _scale_uv_to_mesh_edges(
    uv: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    edges = mesh_edges_from_faces(faces)
    xyz_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    uv_lengths = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)
    valid = uv_lengths > EPS
    if not np.any(valid):
        return uv.copy()
    scale = float(np.median(xyz_lengths[valid] / uv_lengths[valid]))
    return uv * scale


def _gradient_operator(faces: np.ndarray, gradients: np.ndarray, areas: np.ndarray, vertex_count: int):
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    for face_index, tri in enumerate(faces):
        weight = math.sqrt(float(areas[face_index]))
        for axis in range(2):
            row = 2 * face_index + axis
            for local_index, vertex in enumerate(tri):
                rows.append(row)
                cols.append(int(vertex))
                values.append(weight * float(gradients[face_index, axis, local_index]))
    return coo_matrix(
        (values, (rows, cols)), shape=(2 * len(faces), vertex_count)
    ).tocsr()


def _solve_scalar_with_pins(operator, target: np.ndarray, pins: np.ndarray, values: np.ndarray) -> np.ndarray:
    count = operator.shape[1]
    free_mask = np.ones(count, dtype=bool)
    free_mask[pins] = False
    free = np.flatnonzero(free_mask)
    rhs = target - operator[:, pins] @ values
    solution = lsqr(
        operator[:, free], rhs, atol=1.0e-10, btol=1.0e-10,
        iter_lim=max(2000, 2 * count),
    )[0]
    result = np.empty(count, dtype=float)
    result[pins] = values
    result[free] = solution
    return result


def optimize_arap(
    mesh: trimesh.Trimesh,
    initial_uv: np.ndarray,
    locked_vertices: Tuple[int, int],
    iterations: int = 20,
) -> Tuple[np.ndarray, float, int, int]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    _, gradients, areas = _triangle_local_data(vertices, faces)
    operator = _gradient_operator(faces, gradients, areas, len(vertices))
    uv = _scale_uv_to_mesh_edges(initial_uv, vertices, faces)

    signed = _uv_signed_double_areas(uv, faces)
    if float(np.median(signed)) < 0.0:
        uv[:, 1] *= -1.0
    # LSCM needs two pins to remove similarity ambiguity.  ARAP's scalar
    # global systems only have a translation nullspace, so one anchor is
    # sufficient.  Keeping both LSCM pins fixed concentrates otherwise smooth
    # global relaxation into the one-ring of the second pin and creates a
    # visible spike there.
    pins = np.asarray(locked_vertices[:1], dtype=np.int64)
    pin_values = uv[pins].copy()
    completed = 0
    energy_rms = float("inf")

    for iteration in range(max(0, int(iterations))):
        rotations = np.empty((len(faces), 2, 2), dtype=float)
        for face_index, tri in enumerate(faces):
            jacobian = np.asarray([
                gradients[face_index] @ uv[tri, 0],
                gradients[face_index] @ uv[tri, 1],
            ])
            left, _, right_t = np.linalg.svd(jacobian)
            rotation = left @ right_t
            if np.linalg.det(rotation) < 0.0:
                left[:, -1] *= -1.0
                rotation = left @ right_t
            rotations[face_index] = rotation

        target_u = (np.sqrt(areas)[:, None] * rotations[:, 0, :]).reshape(-1)
        target_v = (np.sqrt(areas)[:, None] * rotations[:, 1, :]).reshape(-1)
        candidate = np.column_stack([
            _solve_scalar_with_pins(operator, target_u, pins, pin_values[:, 0]),
            _solve_scalar_with_pins(operator, target_v, pins, pin_values[:, 1]),
        ])

        # Backtrack until every previously valid triangle remains valid.
        valid_before = _uv_signed_double_areas(uv, faces) > EPS
        step = 1.0
        accepted = uv
        while step >= 1.0e-5:
            trial = uv + step * (candidate - uv)
            trial_signed = _uv_signed_double_areas(trial, faces)
            if np.all(trial_signed[valid_before] > EPS):
                accepted = trial
                break
            step *= 0.5
        change = float(np.linalg.norm(accepted - uv) / max(np.linalg.norm(uv), EPS))
        uv = accepted
        completed = iteration + 1

        grad_u = operator @ uv[:, 0]
        grad_v = operator @ uv[:, 1]
        residual = np.concatenate([grad_u - target_u, grad_v - target_v])
        energy_rms = float(np.sqrt(np.mean(residual * residual)))
        if change < 1.0e-8:
            break

    flipped = int(np.count_nonzero(_uv_signed_double_areas(uv, faces) <= EPS))
    return uv, energy_rms, completed, flipped


def gpu_is_available() -> Tuple[bool, str]:
    if cp is None:
        return False, "CuPy is not installed"
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return False, "CuPy found no CUDA device"
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return True, str(name)
    except Exception as exc:
        return False, f"CUDA initialization failed: {exc}"


def _scipy_csr_to_gpu(matrix):
    csr = matrix.tocsr()
    return gpu_csr_matrix(
        (
            cp.asarray(csr.data),
            cp.asarray(csr.indices),
            cp.asarray(csr.indptr),
        ),
        shape=csr.shape,
    )


def _gpu_signed_double_areas(uv, faces):
    triangles = uv[faces]
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    return edge0[:, 0] * edge1[:, 1] - edge0[:, 1] * edge1[:, 0]


def optimize_arap_gpu(
    mesh: trimesh.Trimesh,
    initial_uv: np.ndarray,
    locked_vertices: Tuple[int, int],
    iterations: int = 20,
) -> Tuple[np.ndarray, float, int, int]:
    available, reason = gpu_is_available()
    if not available:
        raise RuntimeError(reason)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    _, gradients, areas = _triangle_local_data(vertices, faces)
    operator_cpu = _gradient_operator(faces, gradients, areas, len(vertices))
    uv_cpu = _scale_uv_to_mesh_edges(initial_uv, vertices, faces)
    if float(np.median(_uv_signed_double_areas(uv_cpu, faces))) < 0.0:
        uv_cpu[:, 1] *= -1.0

    # One anchor removes the translation nullspace without over-constraining
    # the second LSCM pin; see the CPU implementation above.
    pins_cpu = np.asarray(locked_vertices[:1], dtype=np.int64)
    free_mask = np.ones(len(vertices), dtype=bool)
    free_mask[pins_cpu] = False
    free_cpu = np.flatnonzero(free_mask)

    faces_gpu = cp.asarray(faces)
    gradients_gpu = cp.asarray(gradients)
    areas_gpu = cp.asarray(areas)
    sqrt_areas_gpu = cp.sqrt(areas_gpu)
    uv = cp.asarray(uv_cpu)
    pins = cp.asarray(pins_cpu)
    free = cp.asarray(free_cpu)
    pin_values = uv[pins].copy()

    operator = _scipy_csr_to_gpu(operator_cpu)
    free_operator = operator[:, free]
    pin_operator = operator[:, pins]
    free_operator_t = free_operator.T.tocsr()
    normal_matrix = (free_operator_t @ free_operator).tocsc()
    normal_solver = gpu_factorized(normal_matrix)
    completed = 0
    energy_rms = float("inf")

    for iteration in range(max(0, int(iterations))):
        triangle_uv = uv[faces_gpu]
        jacobians = cp.einsum("fak,fkc->fca", gradients_gpu, triangle_uv)
        left, _, right_t = cp.linalg.svd(jacobians)
        rotations = left @ right_t
        reflected = cp.linalg.det(rotations) < 0.0
        if bool(cp.any(reflected).item()):
            left[reflected, :, -1] *= -1.0
            rotations = left @ right_t

        target_u = (sqrt_areas_gpu[:, None] * rotations[:, 0, :]).reshape(-1)
        target_v = (sqrt_areas_gpu[:, None] * rotations[:, 1, :]).reshape(-1)
        rhs_u = target_u - pin_operator @ pin_values[:, 0]
        rhs_v = target_v - pin_operator @ pin_values[:, 1]
        # A is fixed for every ARAP iteration. Factor A.T @ A once, then only
        # update A.T @ b and perform the two triangular solves on the GPU.
        free_u = normal_solver(free_operator_t @ rhs_u)
        free_v = normal_solver(free_operator_t @ rhs_v)
        candidate = cp.empty_like(uv)
        candidate[pins] = pin_values
        candidate[free, 0] = free_u
        candidate[free, 1] = free_v

        valid_before = _gpu_signed_double_areas(uv, faces_gpu) > EPS
        step = 1.0
        accepted = uv
        while step >= 1.0e-5:
            trial = uv + step * (candidate - uv)
            trial_signed = _gpu_signed_double_areas(trial, faces_gpu)
            if bool(cp.all(trial_signed[valid_before] > EPS).item()):
                accepted = trial
                break
            step *= 0.5
        change = float(
            (cp.linalg.norm(accepted - uv) / cp.maximum(cp.linalg.norm(uv), EPS)).item()
        )
        uv = accepted
        completed = iteration + 1

        residual_u = operator @ uv[:, 0] - target_u
        residual_v = operator @ uv[:, 1] - target_v
        energy_rms = float(
            cp.sqrt(
                (cp.sum(residual_u * residual_u) + cp.sum(residual_v * residual_v))
                / (residual_u.size + residual_v.size)
            ).item()
        )
        if change < 1.0e-8:
            break

    flipped = int(cp.count_nonzero(_gpu_signed_double_areas(uv, faces_gpu) <= EPS).item())
    result = cp.asnumpy(uv)
    cp.get_default_memory_pool().free_all_blocks()
    return result, energy_rms, completed, flipped


def edge_stretch_statistics(
    mesh: trimesh.Trimesh, uv: np.ndarray
) -> Tuple[float, float]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = mesh_edges_from_faces(np.asarray(mesh.faces, dtype=np.int64))
    xyz_length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    uv_length = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)
    valid = xyz_length > EPS
    relative = np.abs(uv_length[valid] / xyz_length[valid] - 1.0)
    if len(relative) == 0:
        return 0.0, 0.0
    return float(np.mean(relative)), float(np.max(relative))


def solve_lscm_arap(
    mesh: trimesh.Trimesh,
    iterations: int = 20,
    prefer_gpu: bool = True,
) -> LSCMResult:
    result = solve_lscm(mesh)
    available, device_name = gpu_is_available()
    if prefer_gpu and available:
        uv, energy, completed, flipped = optimize_arap_gpu(
            mesh, result.uv, result.locked_vertices, iterations=iterations
        )
        result.compute_backend = f"GPU ({device_name})"
    else:
        uv, energy, completed, flipped = optimize_arap(
            mesh, result.uv, result.locked_vertices, iterations=iterations
        )
        result.compute_backend = "CPU" if not prefer_gpu else f"CPU fallback ({device_name})"
    mean_stretch, max_stretch = edge_stretch_statistics(mesh, uv)
    result.uv = uv
    result.arap_energy_rms = energy
    result.arap_iterations = completed
    result.flipped_faces = flipped
    result.mean_edge_stretch = mean_stretch
    result.max_edge_stretch = max_stretch
    return result


def point_in_triangle_cross(point: np.ndarray, triangle: np.ndarray, tol: float = 1.0e-10) -> bool:
    a, b, c = triangle
    cross1 = _cross2d(a - point, b - point)
    cross2 = _cross2d(b - point, c - point)
    cross3 = _cross2d(c - point, a - point)
    return (
        (cross1 >= -tol and cross2 >= -tol and cross3 >= -tol)
        or (cross1 <= tol and cross2 <= tol and cross3 <= tol)
    )


def barycentric_area_coordinates(point: np.ndarray, triangle: np.ndarray) -> Optional[np.ndarray]:
    a, b, c = triangle
    total = _cross2d(b - a, c - a)
    if abs(total) < EPS:
        return None
    lambda0 = _cross2d(b - point, c - point) / total
    lambda1 = _cross2d(c - point, a - point) / total
    lambda2 = _cross2d(a - point, b - point) / total
    lambdas = np.asarray([lambda0, lambda1, lambda2], dtype=float)
    return lambdas / np.sum(lambdas)


def find_containing_face(
    point_uv: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    face_tree: cKDTree,
    face_centers: np.ndarray,
    initial_neighbors: int = 3,
) -> Optional[Tuple[int, np.ndarray]]:
    face_count = len(faces)
    neighbor_count = min(max(initial_neighbors, 3), face_count)

    while neighbor_count <= face_count:
        _, nearest_faces = face_tree.query(point_uv, k=neighbor_count)
        nearest_faces = np.atleast_1d(nearest_faces).astype(int)
        order = np.argsort(np.linalg.norm(face_centers[nearest_faces] - point_uv, axis=1))
        for face_index in nearest_faces[order]:
            tri_vertices = faces[face_index]
            triangle_uv = uv[tri_vertices]
            if point_in_triangle_cross(point_uv, triangle_uv):
                lambdas = barycentric_area_coordinates(point_uv, triangle_uv)
                if lambdas is not None and np.all(lambdas >= -1.0e-8):
                    return int(face_index), lambdas

        if neighbor_count == face_count:
            break
        neighbor_count = min(face_count, neighbor_count * 2)
    return None


def inverse_map_uv_paths_to_3d(
    uv_paths: Sequence[np.ndarray],
    uv: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    face_centers = np.mean(uv[faces], axis=1)
    face_tree = cKDTree(face_centers)

    valid_uv_paths = []
    xyz_paths = []
    for path in uv_paths:
        uv_segment = []
        xyz_segment = []
        for point_uv in path:
            found = find_containing_face(point_uv, uv, faces, face_tree, face_centers)
            if found is None:
                if len(xyz_segment) >= 2:
                    valid_uv_paths.append(np.asarray(uv_segment, dtype=float))
                    xyz_paths.append(np.asarray(xyz_segment, dtype=float))
                uv_segment = []
                xyz_segment = []
                continue
            face_index, lambdas = found
            tri = faces[face_index]
            uv_segment.append(point_uv)
            xyz_segment.append(lambdas @ vertices[tri])
        if len(xyz_segment) >= 2:
            valid_uv_paths.append(np.asarray(uv_segment, dtype=float))
            xyz_paths.append(np.asarray(xyz_segment, dtype=float))
    return valid_uv_paths, xyz_paths


def path_parameter_samples(count: int) -> np.ndarray:
    if count <= 1:
        return np.zeros(1, dtype=float)
    return np.arange(count, dtype=float) / float(count - 1)


def fit_bspline_path(points: np.ndarray, sample_count: int) -> np.ndarray:
    if len(points) < 4:
        return points.copy()
    degree = min(3, len(points) - 1)
    try:
        tck, _ = splprep(points.T, s=0.0, k=degree)
        sampled = np.asarray(splev(path_parameter_samples(sample_count), tck), dtype=float).T
        return sampled
    except Exception:
        return points.copy()


def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length2 = float(np.dot(segment, segment))
    if length2 < EPS:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length2, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return points.copy()
    distances = np.asarray([point_line_distance(p, points[0], points[-1]) for p in points[1:-1]], dtype=float)
    if len(distances) == 0:
        return points[[0, -1]]
    max_index = int(np.argmax(distances)) + 1
    if float(distances[max_index - 1]) <= tolerance:
        return points[[0, -1]]
    left = douglas_peucker(points[: max_index + 1], tolerance)
    right = douglas_peucker(points[max_index:], tolerance)
    return np.vstack([left[:-1], right])


def refine_path_by_bspline_and_douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    fitted = fit_bspline_path(points, max(len(points) * 3, 32))
    return douglas_peucker(fitted, tolerance)


def generate_parallel_uv_paths(
    uv: np.ndarray,
    boundary_vertices: np.ndarray,
    line_count: int = 20,
    points_per_line: int = 120,
) -> Tuple[List[np.ndarray], float]:
    if len(boundary_vertices) < 3:
        raise ValueError("Path planning requires boundary vertices PBOU in the UV domain.")

    uv_boundary = uv[boundary_vertices]
    x_min = float(np.min(uv_boundary[:, 0]))
    x_max = float(np.max(uv_boundary[:, 0]))
    y_min = float(np.min(uv_boundary[:, 1]))
    y_max = float(np.max(uv_boundary[:, 1]))
    if abs(y_max - y_min) < EPS:
        raise ValueError("The UV boundary has near-zero height; cannot generate parallel paths.")

    y_values = np.linspace(y_min, y_max, max(2, int(line_count)))
    spacing = float((y_max - y_min) / max(1, len(y_values) - 1))
    paths: List[np.ndarray] = []
    for index, y_value in enumerate(y_values):
        xs = np.linspace(x_min, x_max, int(points_per_line))
        path = np.column_stack([xs, np.full_like(xs, float(y_value))])
        if index % 2 == 1:
            path = path[::-1]
        paths.append(path)
    return paths, spacing


def plan_parallel_paths(
    mesh: trimesh.Trimesh,
    result: LSCMResult,
    line_count: int = 20,
    points_per_line: int = 120,
    douglas_tolerance: Optional[float] = None,
) -> PathPlanningResult:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv_paths, spacing = generate_parallel_uv_paths(result.uv, result.boundary_vertices, line_count, points_per_line)
    valid_uv_paths, xyz_paths = inverse_map_uv_paths_to_3d(uv_paths, result.uv, vertices, faces)

    if douglas_tolerance is None:
        bbox = np.linalg.norm(np.ptp(vertices, axis=0))
        douglas_tolerance = max(float(bbox) * 0.001, 1.0e-6)
    xyz_paths = [refine_path_by_bspline_and_douglas_peucker(path, float(douglas_tolerance)) for path in xyz_paths]
    return PathPlanningResult(
        uv_paths=valid_uv_paths,
        xyz_paths=xyz_paths,
        line_count=len(xyz_paths),
        path_spacing=spacing,
    )


def set_axes_equal_3d(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius < EPS:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_lscm_result(
    mesh: trimesh.Trimesh,
    result: LSCMResult,
    mesh_path: Path,
    paths: Optional[PathPlanningResult] = None,
):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = result.uv
    edges = mesh_edges_from_faces(faces)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"LSCM + ARAP UV parameterization - {mesh_path.name}", fontsize=13)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    max_3d_faces = 20000
    if len(faces) > max_3d_faces:
        sample = np.linspace(0, len(faces) - 1, max_3d_faces, dtype=np.int64)
        face_vertices = vertices[faces[sample]]
    else:
        face_vertices = vertices[faces]
    collection = Poly3DCollection(face_vertices, facecolor="#86a9c8", edgecolor="#243746", linewidths=0.15, alpha=0.88)
    ax3d.add_collection3d(collection)
    if paths is not None:
        for path in paths.xyz_paths:
            ax3d.plot(path[:, 0], path[:, 1], path[:, 2], color="#d62828", linewidth=1.2)
    locked = vertices[list(result.locked_vertices)]
    ax3d.scatter(locked[:, 0], locked[:, 1], locked[:, 2], c=["#d62828", "#f77f00"], s=45, depthshade=False)
    if result.seam_vertices is not None and len(result.seam_vertices) > 0:
        seam_points = vertices[result.seam_vertices]
        ax3d.scatter(
            seam_points[:, 0], seam_points[:, 1], seam_points[:, 2],
            c="#e63946", s=14, depthshade=False, label="cut seam",
        )
    ax3d.set_title("3D triangular mesh")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    set_axes_equal_3d(ax3d, vertices)

    axuv = fig.add_subplot(1, 2, 2)
    segments = uv[edges]
    line_collection = LineCollection(segments, colors="#1f2933", linewidths=0.35, alpha=0.9)
    axuv.add_collection(line_collection)
    if len(result.boundary_vertices) > 0:
        axuv.scatter(uv[result.boundary_vertices, 0], uv[result.boundary_vertices, 1], s=4, c="#2a9d8f", label="boundary")
    if paths is not None:
        for path in paths.uv_paths:
            axuv.plot(path[:, 0], path[:, 1], color="#d62828", linewidth=0.9)
    axuv.scatter(
        uv[list(result.locked_vertices), 0],
        uv[list(result.locked_vertices), 1],
        c=["#d62828", "#f77f00"],
        s=45,
        label="LSCM pins (red = ARAP anchor)",
        zorder=3,
    )
    axuv.set_title("2D UV parameter domain after ARAP")
    axuv.set_xlabel("u")
    axuv.set_ylabel("v")
    axuv.set_aspect("equal", adjustable="datalim")
    axuv.autoscale()
    axuv.grid(True, color="#d0d7de", linewidth=0.5)
    axuv.legend(loc="best", fontsize=8)

    lines = [
        f"vertices: {len(vertices)}",
        f"faces: {len(faces)}",
        f"boundary vertices: {len(result.boundary_vertices)}",
        f"LSCM pins: {result.locked_vertices[0]}, {result.locked_vertices[1]}",
        f"ARAP anchor: {result.locked_vertices[0]}",
        f"LSCM residual RMS: {result.energy_rms:.3e}",
        f"ARAP backend: {result.compute_backend}",
        f"ARAP iterations / RMS: {result.arap_iterations} / {result.arap_energy_rms:.3e}",
        f"flipped UV faces: {result.flipped_faces}",
        f"mean / max edge stretch error: {result.mean_edge_stretch:.3%} / {result.max_edge_stretch:.3%}",
    ]
    if paths is not None:
        lines.extend([
            f"path lines: {paths.line_count}",
            f"UV line spacing: {paths.path_spacing:.6g}",
        ])
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.show()


def export_uv_csv(path: str | Path, mesh: trimesh.Trimesh, result: LSCMResult) -> None:
    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["vertex_index", "u", "v", "x", "y", "z"])
        for index, (point, uv) in enumerate(zip(mesh.vertices, result.uv)):
            writer.writerow([index, uv[0], uv[1], point[0], point[1], point[2]])


def export_textured_obj(path: str | Path, mesh: trimesh.Trimesh, uv: np.ndarray) -> None:
    output = Path(path)
    uv_min = uv.min(axis=0)
    # One common scale preserves the ARAP aspect ratio and length ratios.
    scale = max(float(np.max(np.ptp(uv, axis=0))), EPS)
    normalized = (uv - uv_min) / scale
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# LSCM + ARAP parameterized mesh\n")
        for vertex in np.asarray(mesh.vertices):
            stream.write(f"v {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}\n")
        for coord in normalized:
            stream.write(f"vt {coord[0]:.12g} {coord[1]:.12g}\n")
        for face in np.asarray(mesh.faces, dtype=np.int64) + 1:
            stream.write("f " + " ".join(f"{i}/{i}" for i in face) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut two-boundary lofts and parameterize meshes with LSCM + ARAP."
    )
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY mesh")
    parser.add_argument("--cpu", action="store_true", help="Force CPU ARAP instead of CUDA")
    parser.add_argument(
        "--arap-iterations", type=int, default=20,
        help="Maximum local-global ARAP iterations",
    )
    parser.add_argument(
        "--plan-paths", action="store_true",
        help="Plan and inverse-map parallel UV paths after parameterization",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    mesh_path = Path(args.mesh) if args.mesh else select_mesh_file_dialog()
    if mesh_path is None:
        print("No mesh selected.")
        return

    print(f"Loading mesh: {mesh_path}")
    source_mesh = load_mesh(mesh_path)
    prepared = prepare_parameterization_mesh(source_mesh)
    mesh = prepared.mesh
    print(f"Boundary loops before preparation: {prepared.boundary_count_before}")
    if prepared.boundary_count_before == 2:
        print(
            "Two-boundary loft detected: cut along the loft reference curve "
            f"({len(prepared.seam_points)} seam vertices)."
        )
        print(f"Boundary loops after cutting: {prepared.boundary_count_after}")
    elif prepared.boundary_count_before == 0:
        raise ValueError("A closed mesh has no boundary. Define a seam before LSCM parameterization.")
    print(f"Solving LSCM + ARAP for {len(mesh.vertices)} vertices and {len(mesh.faces)} faces...")
    available, gpu_status = gpu_is_available()
    print(f"CUDA status: {'available' if available else 'unavailable'} ({gpu_status})")
    result = solve_lscm_arap(
        mesh,
        iterations=max(0, args.arap_iterations),
        prefer_gpu=not args.cpu,
    )
    result.seam_vertices = prepared.seam_vertices
    print(f"Boundary vertices: {len(result.boundary_vertices)}")
    print(f"LSCM locked vertices: {result.locked_vertices}")
    print(f"ARAP anchor vertex: {result.locked_vertices[0]}")
    print(f"LSCM residual RMS: {result.energy_rms:.6e}")
    print(f"ARAP iterations: {result.arap_iterations}")
    print(f"ARAP backend: {result.compute_backend}")
    print(f"ARAP residual RMS: {result.arap_energy_rms:.6e}")
    print(f"Flipped UV faces: {result.flipped_faces}")
    print(f"Mean edge stretch error: {result.mean_edge_stretch:.3%}")
    print(f"Maximum edge stretch error: {result.max_edge_stretch:.3%}")
    if result.flipped_faces > 0:
        print(
            "WARNING: The UV map contains flipped faces and is not yet safe "
            "for final texture mapping."
        )
    output_prefix = mesh_path.with_name(mesh_path.stem + "_lscm_arap")
    csv_path = output_prefix.with_suffix(".csv")
    obj_path = output_prefix.with_suffix(".obj")
    export_uv_csv(csv_path, mesh, result)
    export_textured_obj(obj_path, mesh, result.uv)
    print(f"Saved UV CSV: {csv_path}")
    print(f"Saved textured OBJ: {obj_path}")
    paths = None
    if args.plan_paths:
        print("Planning 20 parallel equidistant paths in the UV plane...")
        paths = plan_parallel_paths(mesh, result, line_count=20, points_per_line=60)
        print(f"Mapped path lines: {paths.line_count}")
        print(f"UV line spacing: {paths.path_spacing:.6g}")
    plot_lscm_result(mesh, result, mesh_path, paths)


if __name__ == "__main__":
    main()
