#!/usr/bin/env python3
"""Basic Least-Squares Conformal Maps (LSCM) parameterization.

This is the deliberately small counterpart of ``LSCMARAP.py``.  It performs
only the LSCM initialization described in Section 2.2 of the paper: no ARAP,
no CUDA/CuPy, and no path planning.

Meshes with one boundary are parameterized directly.  A connected genus-zero
mesh with multiple boundaries is cut along an automatically generated seam
tree so that LSCM receives a disk-like surface.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr


EPS = 1.0e-12


@dataclass
class LSCMResult:
    uv: np.ndarray
    locked_vertices: Tuple[int, int]
    boundary_vertices: np.ndarray
    energy_rms: float
    flipped_faces: int
    mean_edge_stretch: float
    max_edge_stretch: float
    seam_vertices: Optional[np.ndarray] = None


@dataclass
class PreparedMesh:
    mesh: trimesh.Trimesh
    seam_vertices: np.ndarray
    boundary_count_before: int
    boundary_count_after: int


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
        title="Select STL/OBJ/PLY mesh for basic LSCM",
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
        geometries = [
            geometry for geometry in mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"No triangular mesh found in {path}.")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The selected mesh is empty.")

    mesh = mesh.copy()
    mesh.merge_vertices()
    if hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def boundary_vertices_from_faces(
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges.astype(np.int64, copy=False), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return np.empty(0, dtype=np.int64), boundary_edges.reshape(0, 2)
    return np.unique(boundary_edges), boundary_edges


def ordered_boundary_loops(faces: np.ndarray) -> List[np.ndarray]:
    _, boundary_edges = boundary_vertices_from_faces(faces)
    if len(boundary_edges) == 0:
        return []
    adjacency: dict[int, List[int]] = {}
    unused: set[Tuple[int, int]] = set()
    for a_raw, b_raw in boundary_edges:
        a, b = int(a_raw), int(b_raw)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        unused.add((min(a, b), max(a, b)))

    loops: List[np.ndarray] = []
    while unused:
        start, current = next(iter(unused))
        unused.remove((start, current))
        loop = [start, current]
        previous = start
        while current != start:
            next_vertex = None
            for candidate in adjacency[current]:
                if candidate == previous:
                    continue
                edge = (min(current, candidate), max(current, candidate))
                if edge in unused:
                    unused.remove(edge)
                    next_vertex = candidate
                    break
            if next_vertex is None:
                raise ValueError("The mesh boundary is open, branched, or non-manifold.")
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges.astype(np.int64, copy=False), axis=1), axis=0)


def _nearest_boundary_pair(
    vertices: np.ndarray, first: np.ndarray, second: np.ndarray
) -> Tuple[int, int]:
    # Boundary loops are normally small; chunking avoids a large temporary
    # matrix for unusually dense meshes.
    best_distance = float("inf")
    best_pair = (int(first[0]), int(second[0]))
    for start in range(0, len(first), 512):
        indices = first[start:start + 512]
        delta = vertices[indices, None, :] - vertices[second][None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        flat = int(np.argmin(distance2))
        row, column = np.unravel_index(flat, distance2.shape)
        if float(distance2[row, column]) < best_distance:
            best_distance = float(distance2[row, column])
            best_pair = (int(indices[row]), int(second[column]))
    return best_pair


def _shortest_mesh_path(mesh: trimesh.Trimesh, start: int, end: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = mesh_edges_from_faces(np.asarray(mesh.faces, dtype=np.int64))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = coo_matrix(
        (
            np.concatenate([lengths, lengths]),
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
        raise ValueError("Could not find a seam path between the two boundaries.")
    path = [int(end)]
    while path[-1] != start:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("The seam predecessor chain is incomplete.")
        path.append(predecessor)
    return np.asarray(path[::-1], dtype=np.int64)


def _shortest_path_to_other_boundary(
    mesh: trimesh.Trimesh,
    source_loop: np.ndarray,
    other_loops: Sequence[np.ndarray],
) -> np.ndarray:
    """Find an interior path from one boundary to the closest other boundary.

    Non-source boundary vertices are terminals: the directed graph permits a
    path to enter them but not leave them.  Consequently, a seam cannot travel
    along or pass through a third boundary on its way to the selected target.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = mesh_edges_from_faces(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    source_set = set(map(int, source_loop))
    terminal_set = set(map(int, np.concatenate(other_loops)))
    rows: List[int] = []
    columns: List[int] = []
    weights: List[float] = []
    for (a_raw, b_raw), length in zip(edges, lengths):
        a, b = int(a_raw), int(b_raw)
        if a not in terminal_set or a in source_set:
            rows.append(a)
            columns.append(b)
            weights.append(float(length))
        if b not in terminal_set or b in source_set:
            rows.append(b)
            columns.append(a)
            weights.append(float(length))

    super_source = len(vertices)
    positive_lengths = lengths[lengths > EPS]
    connector_weight = (
        max(float(np.median(positive_lengths)), 1.0) * 1.0e-9
        if len(positive_lengths) else 1.0e-9
    )
    for vertex in source_set:
        rows.append(super_source)
        columns.append(vertex)
        weights.append(connector_weight)
    graph = coo_matrix(
        (weights, (rows, columns)),
        shape=(len(vertices) + 1, len(vertices) + 1),
    ).tocsr()
    distance, predecessors = dijkstra(
        graph, directed=True, indices=super_source, return_predecessors=True
    )
    target = min(terminal_set, key=lambda vertex: float(distance[vertex]))
    if not np.isfinite(distance[target]):
        raise ValueError("Could not connect the remaining mesh boundaries by an interior seam.")
    path = [int(target)]
    while path[-1] != super_source:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("The multi-boundary seam predecessor chain is incomplete.")
        path.append(predecessor)
    path.reverse()
    return np.asarray(path[1:], dtype=np.int64)


def _faces_to_duplicate_along_seam(
    faces: np.ndarray, seam: np.ndarray
) -> dict[int, set[int]]:
    if len(seam) < 2 or len(np.unique(seam)) != len(seam):
        raise ValueError("The seam must be a simple path with at least two vertices.")
    seam_edges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in zip(seam[:-1], seam[1:])
    }
    incident: dict[int, List[int]] = {int(vertex): [] for vertex in seam}
    seam_edge_faces: dict[Tuple[int, int], List[int]] = {}
    for face_index, tri_raw in enumerate(faces):
        tri = [int(vertex) for vertex in tri_raw]
        for vertex in tri:
            if vertex in incident:
                incident[vertex].append(face_index)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (min(a, b), max(a, b))
            if edge in seam_edges:
                seam_edge_faces.setdefault(edge, []).append(face_index)

    components: dict[int, List[set[int]]] = {}
    for vertex in map(int, seam):
        adjacency = {face_index: set() for face_index in incident[vertex]}
        local_edge_faces: dict[Tuple[int, int], List[int]] = {}
        for face_index in incident[vertex]:
            for other_raw in faces[face_index]:
                other = int(other_raw)
                if other == vertex:
                    continue
                edge = (min(vertex, other), max(vertex, other))
                if edge not in seam_edges:
                    local_edge_faces.setdefault(edge, []).append(face_index)
        for shared_faces in local_edge_faces.values():
            for face_index in shared_faces:
                adjacency[face_index].update(set(shared_faces) - {face_index})

        remaining = set(incident[vertex])
        local_components: List[set[int]] = []
        while remaining:
            stack = [remaining.pop()]
            component: set[int] = set()
            while stack:
                face_index = stack.pop()
                component.add(face_index)
                neighbours = adjacency[face_index] & remaining
                remaining.difference_update(neighbours)
                stack.extend(neighbours)
            local_components.append(component)
        if len(local_components) != 2:
            raise ValueError(
                f"Seam vertex {vertex} is not locally manifold/cuttable "
                f"(found {len(local_components)} face fans)."
            )
        components[vertex] = local_components

    selected: dict[int, set[int]] = {}
    for edge_index, (a_raw, b_raw) in enumerate(zip(seam[:-1], seam[1:])):
        a, b = int(a_raw), int(b_raw)
        edge = (min(a, b), max(a, b))
        adjacent = seam_edge_faces.get(edge, [])
        if len(adjacent) != 2:
            raise ValueError(
                f"Seam edge {edge} must have two adjacent faces, found {len(adjacent)}."
            )
        if edge_index == 0:
            selected[a] = next(part for part in components[a] if adjacent[0] in part)
        selected_face = next(face for face in adjacent if face in selected[a])
        selected[b] = next(part for part in components[b] if selected_face in part)
    return selected


def _cut_along_path(
    mesh: trimesh.Trimesh, seam: np.ndarray
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    duplicate_faces = _faces_to_duplicate_along_seam(faces, seam)
    duplicate_map = {
        int(vertex): len(vertices) + index for index, vertex in enumerate(seam)
    }
    cut_faces = faces.copy()
    for face_index, tri in enumerate(cut_faces):
        for local_index, vertex_raw in enumerate(tri):
            vertex = int(vertex_raw)
            if vertex in duplicate_faces and face_index in duplicate_faces[vertex]:
                tri[local_index] = duplicate_map[vertex]
    cut_vertices = np.vstack([vertices, vertices[seam]])
    cut_mesh = trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces, process=False)
    # Both the original seam vertices and their appended copies are visible.
    seam_vertices = np.concatenate([
        seam,
        np.arange(len(vertices), len(vertices) + len(seam), dtype=np.int64),
    ])
    return cut_mesh, seam_vertices


def cut_multiple_boundary_mesh(mesh: trimesh.Trimesh) -> PreparedMesh:
    """Connect every boundary to one growing outer boundary by a cut tree."""
    boundary_count_before = len(
        ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    )
    if boundary_count_before < 2:
        raise ValueError(
            f"Multi-boundary cutting requires at least two boundaries, "
            f"found {boundary_count_before}."
        )
    cut_mesh = mesh
    all_seam_vertices: List[np.ndarray] = []
    previous_count = boundary_count_before
    while previous_count > 1:
        loops = ordered_boundary_loops(
            np.asarray(cut_mesh.faces, dtype=np.int64)
        )
        # After each cut this loop contains the previously merged boundaries
        # and normally becomes the longest loop.  Growing one cut tree avoids
        # disconnected seams and requires only one Dijkstra solve per hole.
        source_index = int(np.argmax([len(loop) for loop in loops]))
        source_loop = loops[source_index]
        other_loops = [loop for index, loop in enumerate(loops) if index != source_index]
        seam = _shortest_path_to_other_boundary(cut_mesh, source_loop, other_loops)
        cut_mesh, seam_vertices = _cut_along_path(cut_mesh, seam)
        all_seam_vertices.append(seam_vertices)
        current_count = len(
            ordered_boundary_loops(np.asarray(cut_mesh.faces, dtype=np.int64))
        )
        if current_count != previous_count - 1:
            raise ValueError(
                "An automatic seam did not merge exactly two boundaries "
                f"({previous_count} -> {current_count})."
            )
        previous_count = current_count
    return PreparedMesh(
        mesh=cut_mesh,
        seam_vertices=np.unique(np.concatenate(all_seam_vertices)),
        boundary_count_before=boundary_count_before,
        boundary_count_after=1,
    )


def prepare_parameterization_mesh(mesh: trimesh.Trimesh) -> PreparedMesh:
    count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
    if count == 0:
        raise ValueError("A closed mesh needs a user-defined seam before LSCM.")
    if count == 1:
        return PreparedMesh(mesh, np.empty(0, dtype=np.int64), 1, 1)
    return cut_multiple_boundary_mesh(mesh)


def choose_locked_vertices(
    vertices: np.ndarray, boundary_vertices: np.ndarray
) -> Tuple[int, int]:
    candidates = np.asarray(boundary_vertices, dtype=np.int64)
    if len(candidates) < 2:
        raise ValueError("LSCM requires at least two boundary vertices.")
    seed = int(candidates[0])
    first = int(candidates[np.argmax(np.linalg.norm(vertices[candidates] - vertices[seed], axis=1))])
    second = int(candidates[np.argmax(np.linalg.norm(vertices[candidates] - vertices[first], axis=1))])
    if first == second:
        raise ValueError("Cannot find two distinct LSCM pins.")
    return first, second


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < EPS:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vector / length


def local_triangle_coordinates(points: np.ndarray) -> Tuple[np.ndarray, float]:
    p0, p1, p2 = points
    x_axis = _unit(p1 - p0, "triangle edge")
    normal = _unit(np.cross(x_axis, p2 - p0), "triangle normal")
    y_axis = _unit(np.cross(normal, x_axis), "triangle Y axis")
    relative = points - p0
    xy = np.column_stack([relative @ x_axis, relative @ y_axis])
    area = 0.5 * abs(
        (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
        - (xy[1, 1] - xy[0, 1]) * (xy[2, 0] - xy[0, 0])
    )
    if area < EPS:
        raise ValueError("Degenerate triangle with near-zero area.")
    return xy, area


def triangle_gradient_matrix(xy: np.ndarray, area: float) -> np.ndarray:
    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    return np.asarray(
        [[y1 - y2, y2 - y0, y0 - y1],
         [x2 - x1, x0 - x2, x1 - x0]],
        dtype=float,
    ) / (2.0 * area)


def assemble_lscm_system(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[coo_matrix, np.ndarray]:
    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    row = 0
    skipped = 0
    vertex_count = len(vertices)
    for tri in faces:
        try:
            xy, area = local_triangle_coordinates(vertices[tri])
        except ValueError:
            skipped += 1
            continue
        gradient = triangle_gradient_matrix(xy, area)
        dx, dy = gradient[0], gradient[1]
        weight = math.sqrt(area)
        for local_index, vertex_raw in enumerate(tri):
            vertex = int(vertex_raw)
            rows.extend([row, row, row + 1, row + 1])
            columns.extend([vertex, vertex_count + vertex, vertex, vertex_count + vertex])
            values.extend([
                weight * dx[local_index], -weight * dy[local_index],
                weight * dy[local_index], weight * dx[local_index],
            ])
        row += 2
    if row == 0:
        raise ValueError("No non-degenerate triangles are available for LSCM.")
    if skipped:
        warnings.warn(f"Skipped {skipped} degenerate triangles.")
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row, 2 * vertex_count)
    )
    return matrix, np.zeros(row, dtype=float)


def _signed_double_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
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
    return uv * float(np.median(xyz_lengths[valid] / uv_lengths[valid]))


def edge_stretch_statistics(
    vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray
) -> Tuple[float, float]:
    edges = mesh_edges_from_faces(faces)
    xyz_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    uv_lengths = np.linalg.norm(uv[edges[:, 0]] - uv[edges[:, 1]], axis=1)
    valid = xyz_lengths > EPS
    relative = np.abs(uv_lengths[valid] / xyz_lengths[valid] - 1.0)
    if len(relative) == 0:
        return 0.0, 0.0
    return float(np.mean(relative)), float(np.max(relative))


def solve_lscm(mesh: trimesh.Trimesh) -> LSCMResult:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_vertices, _ = boundary_vertices_from_faces(faces)
    lock0, lock1 = choose_locked_vertices(vertices, boundary_vertices)
    matrix, rhs = assemble_lscm_system(vertices, faces)
    vertex_count = len(vertices)
    fixed_values = {
        lock0: 0.0,
        vertex_count + lock0: 0.0,
        lock1: 1.0,
        vertex_count + lock1: 0.0,
    }
    fixed_columns = np.asarray(sorted(fixed_values), dtype=np.int64)
    fixed = np.asarray([fixed_values[int(column)] for column in fixed_columns])
    free_mask = np.ones(2 * vertex_count, dtype=bool)
    free_mask[fixed_columns] = False
    free_columns = np.flatnonzero(free_mask)
    system = matrix.tocsr()
    free_rhs = rhs - system[:, fixed_columns] @ fixed
    free_solution = lsqr(
        system[:, free_columns], free_rhs,
        atol=1.0e-10, btol=1.0e-10,
        iter_lim=max(2000, 2 * vertex_count),
    )[0]
    solution = np.zeros(2 * vertex_count, dtype=float)
    solution[fixed_columns] = fixed
    solution[free_columns] = free_solution
    uv = np.column_stack([solution[:vertex_count], solution[vertex_count:]])
    residual = system @ solution
    energy_rms = float(np.sqrt(np.mean(residual * residual)))
    uv = _scale_uv_to_mesh_edges(uv, vertices, faces)
    if float(np.median(_signed_double_areas(uv, faces))) < 0.0:
        uv[:, 1] *= -1.0
    flipped = int(np.count_nonzero(_signed_double_areas(uv, faces) <= EPS))
    mean_stretch, max_stretch = edge_stretch_statistics(vertices, faces, uv)
    return LSCMResult(
        uv=uv,
        locked_vertices=(lock0, lock1),
        boundary_vertices=boundary_vertices,
        energy_rms=energy_rms,
        flipped_faces=flipped,
        mean_edge_stretch=mean_stretch,
        max_edge_stretch=max_stretch,
    )


def export_uv_csv(path: str | Path, mesh: trimesh.Trimesh, result: LSCMResult) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["vertex_index", "u", "v", "x", "y", "z"])
        for index, (point, uv) in enumerate(zip(mesh.vertices, result.uv)):
            writer.writerow([index, uv[0], uv[1], point[0], point[1], point[2]])


def export_textured_obj(path: str | Path, mesh: trimesh.Trimesh, uv: np.ndarray) -> None:
    uv_min = uv.min(axis=0)
    scale = max(float(np.max(np.ptp(uv, axis=0))), EPS)
    normalized = (uv - uv_min) / scale
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Basic LSCM parameterized mesh\n")
        for vertex in np.asarray(mesh.vertices):
            stream.write(f"v {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}\n")
        for coord in normalized:
            stream.write(f"vt {coord[0]:.12g} {coord[1]:.12g}\n")
        for face in np.asarray(mesh.faces, dtype=np.int64) + 1:
            stream.write("f " + " ".join(f"{index}/{index}" for index in face) + "\n")


def _set_axes_equal_3d(axis, vertices: np.ndarray) -> None:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(0.5 * float(np.max(maximum - minimum)), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_lscm_result(
    mesh: trimesh.Trimesh, result: LSCMResult, mesh_path: Path
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = mesh_edges_from_faces(faces)
    figure = plt.figure(figsize=(14, 7))
    figure.suptitle(f"Basic LSCM UV parameterization - {mesh_path.name}", fontsize=13)

    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    collection = Poly3DCollection(
        vertices[faces], facecolor="#86a9c8", edgecolor="#243746",
        linewidths=0.15, alpha=0.88,
    )
    axis_3d.add_collection3d(collection)
    pins_3d = vertices[list(result.locked_vertices)]
    axis_3d.scatter(
        pins_3d[:, 0], pins_3d[:, 1], pins_3d[:, 2],
        c=["#d62828", "#f77f00"], s=45, depthshade=False,
    )
    if result.seam_vertices is not None and len(result.seam_vertices):
        seam = vertices[result.seam_vertices]
        axis_3d.scatter(seam[:, 0], seam[:, 1], seam[:, 2], c="#e63946", s=10)
    axis_3d.set_title("3D triangular mesh")
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    _set_axes_equal_3d(axis_3d, vertices)

    axis_uv = figure.add_subplot(1, 2, 2)
    axis_uv.add_collection(LineCollection(result.uv[edges], colors="#1f2933", linewidths=0.35))
    axis_uv.scatter(
        result.uv[result.boundary_vertices, 0],
        result.uv[result.boundary_vertices, 1],
        s=4, c="#2a9d8f", label="boundary",
    )
    pins_uv = result.uv[list(result.locked_vertices)]
    axis_uv.scatter(
        pins_uv[:, 0], pins_uv[:, 1], c=["#d62828", "#f77f00"],
        s=45, label="LSCM locked vertices", zorder=3,
    )
    axis_uv.set_title("2D UV parameter domain after LSCM")
    axis_uv.set_xlabel("u")
    axis_uv.set_ylabel("v")
    axis_uv.set_aspect("equal", adjustable="datalim")
    axis_uv.autoscale()
    axis_uv.grid(True, color="#d0d7de", linewidth=0.5)
    axis_uv.legend(loc="best", fontsize=8)
    information = [
        f"vertices: {len(vertices)}",
        f"faces: {len(faces)}",
        f"boundary vertices: {len(result.boundary_vertices)}",
        f"LSCM pins: {result.locked_vertices[0]}, {result.locked_vertices[1]}",
        f"LSCM residual RMS: {result.energy_rms:.3e}",
        f"flipped UV faces: {result.flipped_faces}",
        f"mean / max edge stretch: {result.mean_edge_stretch:.3%} / {result.max_edge_stretch:.3%}",
    ]
    figure.text(0.01, 0.01, "\n".join(information), fontsize=9, va="bottom")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.show()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic LSCM-only UV parameterization (no ARAP or CUDA)."
    )
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY mesh")
    parser.add_argument(
        "--no-plot", action="store_true", help="Export results without opening a plot window"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    mesh_path = Path(args.mesh) if args.mesh else select_mesh_file_dialog()
    if mesh_path is None:
        print("No mesh selected.")
        return
    print(f"Loading mesh: {mesh_path}")
    prepared = prepare_parameterization_mesh(load_mesh(mesh_path))
    mesh = prepared.mesh
    print(f"Boundary loops before preparation: {prepared.boundary_count_before}")
    print(f"Boundary loops used by LSCM: {prepared.boundary_count_after}")
    if prepared.boundary_count_before > 1:
        print(f"Automatic seam vertices: {len(prepared.seam_vertices)}")
    print(f"Solving basic LSCM for {len(mesh.vertices)} vertices and {len(mesh.faces)} faces...")
    result = solve_lscm(mesh)
    result.seam_vertices = prepared.seam_vertices
    print(f"LSCM locked vertices: {result.locked_vertices}")
    print(f"LSCM residual RMS: {result.energy_rms:.6e}")
    print(f"Flipped UV faces: {result.flipped_faces}")
    print(f"Mean edge stretch error: {result.mean_edge_stretch:.3%}")
    print(f"Maximum edge stretch error: {result.max_edge_stretch:.3%}")
    if result.flipped_faces:
        print(
            "WARNING: Basic LSCM produced flipped UV faces. "
            "Use LSCMARAP.py when a flip-free optimized map is required."
        )

    output_prefix = mesh_path.with_name(mesh_path.stem + "_lscm")
    csv_path = output_prefix.with_suffix(".csv")
    obj_path = output_prefix.with_suffix(".obj")
    export_uv_csv(csv_path, mesh, result)
    export_textured_obj(obj_path, mesh, result.uv)
    print(f"Saved UV CSV: {csv_path}")
    print(f"Saved textured OBJ: {obj_path}")
    if not args.no_plot:
        plot_lscm_result(mesh, result, mesh_path)


if __name__ == "__main__":
    main()
