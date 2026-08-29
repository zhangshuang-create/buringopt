from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import uuid
import warnings
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import heapq
import numpy as np
import trimesh
import vtk
from scipy.spatial import cKDTree
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import splprep, splev  # 【必须补上这句】
#超椭圆拟合（还未进入圆弧）
ROOT = Path(__file__).resolve().parents[2]
OPTCUTS_ROOT = ROOT / "third_party" / "OptCuts"
OPTCUTS_EXE = OPTCUTS_ROOT / "build" / "Release" / "OptCuts_bin.exe"
FINAL_FRAME_HOLD_MS = 2000
DEFAULT_TRAJECTORY_SPACING = 30.0
DEFAULT_TRAJECTORY_SPEED = 300.0
DEFAULT_SPRAY_DISTANCE = 0.0
EPS = 1.0e-9


@dataclass(frozen=True)
class PrecutSeam:
    """The two vertex copies created before OptCuts for one physical seam."""

    original_vertex_ids: np.ndarray
    cut_vertex_pairs: np.ndarray
    xyz: np.ndarray

    @property
    def edges(self) -> np.ndarray:
        if len(self.xyz) < 2:
            return np.empty((0, 2, 3), dtype=float)
        return self.xyz[np.column_stack([
            np.arange(len(self.xyz) - 1),
            np.arange(1, len(self.xyz)),
        ])]


@dataclass(frozen=True)
class ObjUVData:
    xyz: np.ndarray
    xyz_faces: np.ndarray
    uv: np.ndarray
    uv_faces: np.ndarray


@dataclass(frozen=True)
class SeamUVSegment:
    """Two UV boundary edges representing one physical pre-cut seam edge."""

    physical_index: int
    uv_a0: np.ndarray
    uv_a1: np.ndarray
    uv_b0: np.ndarray
    uv_b1: np.ndarray


@dataclass
class TrajectorySegment:
    """One explicit path event; UV is absent for off-surface 3D motion."""
    kind: str
    uv: np.ndarray
    xyz: np.ndarray
    spray_on: bool
    note: str = ""
    region: str = ""
    phases: Optional[np.ndarray] = None
    geometry_meta: Optional[dict] = None  # 【新增】：持久化保存解析几何公式与方程参数


@dataclass(frozen=True)
class PlanningFrame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    waist_width: float
    waist_pair_index: int


@dataclass(frozen=True)
class Optimum:
    spacing: float
    counts: np.ndarray
    actual_spacings: np.ndarray
    objective: float
    candidates: np.ndarray
    objective_values: np.ndarray


@dataclass
class BoundaryLoop:
    points: np.ndarray
    vertex_indices: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    z_level: float
    name: str = ""


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray
    d: float


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
    seam_vertices = np.concatenate([
        seam,
        np.arange(len(vertices), len(vertices) + len(seam), dtype=np.int64),
    ])
    return cut_mesh, seam_vertices


def _as_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected an array with shape (N, 3).")
    return arr


def _unit(vec: np.ndarray, name: str = "vector") -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < EPS:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vec / norm


def _polygon_normal(points: np.ndarray) -> np.ndarray:
    pts = _as_array(points)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    normal = np.zeros(3)
    for p0, p1 in zip(pts, np.roll(pts, -1, axis=0)):
        normal[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        normal[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        normal[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    if np.linalg.norm(normal) < EPS:
        return np.array([0.0, 0.0, 1.0])
    return normal / np.linalg.norm(normal)


def _plane_from_three_points(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Plane:
    normal = _unit(np.cross(p2 - p1, p3 - p1), "plane normal")
    return Plane(
        point=np.asarray(p1, dtype=float),
        normal=normal,
        d=-float(np.dot(normal, p1)),
    )


def _deduplicate_points(points: np.ndarray, tol: float = 1.0e-7) -> np.ndarray:
    pts = _as_array(points)
    if len(pts) == 0:
        return pts
    scale = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1.0)
    qtol = max(tol * scale, tol)
    keys = np.round(pts / qtol).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(keep)]


def _arc_lengths(polyline: np.ndarray) -> np.ndarray:
    pts = _as_array(polyline)
    if len(pts) == 0:
        return np.zeros(0)
    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = _unit(normal, "basis normal")
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(normal, helper), "plane basis u")
    second = _unit(np.cross(normal, first), "plane basis v")
    return first, second


def extract_boundary_loops(mesh: trimesh.Trimesh) -> List[BoundaryLoop]:
    """Extract ordered open-boundary loops by counting undirected triangle edges."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    edge_count: dict[Tuple[int, int], int] = {}
    for triangle in faces:
        for index in range(3):
            a, b = int(triangle[index]), int(triangle[(index + 1) % 3])
            if a > b:
                a, b = b, a
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        raise ValueError("No open boundary edges found. The mesh may be closed.")

    adjacency: dict[int, List[int]] = {}
    edge_set = set(boundary_edges)
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited_edges: set[Tuple[int, int]] = set()
    loops: List[BoundaryLoop] = []

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    starts = sorted(adjacency, key=lambda index: (len(adjacency[index]) != 1, index))
    for start in starts:
        has_unvisited = any(
            edge_key(start, neighbor) not in visited_edges
            for neighbor in adjacency[start]
        )
        if not has_unvisited:
            continue

        ordered = [start]
        previous = -1
        current = start
        while True:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
                   and edge_key(current, neighbor) in edge_set
                   and edge_key(current, neighbor) not in visited_edges
            ]
            if not candidates and previous >= 0:
                candidates = [
                    neighbor
                    for neighbor in adjacency[current]
                    if edge_key(current, neighbor) in edge_set
                       and edge_key(current, neighbor) not in visited_edges
                ]
            if not candidates:
                break
            next_vertex = candidates[0]
            visited_edges.add(edge_key(current, next_vertex))
            if next_vertex == start:
                break
            ordered.append(next_vertex)
            previous, current = current, next_vertex
            if len(ordered) > len(boundary_edges) + 2:
                warnings.warn(
                    "Boundary tracing stopped because it exceeded expected edge count."
                )
                break

        points = vertices[np.asarray(ordered, dtype=np.int64)]
        if len(points) < 3:
            continue
        center = points.mean(axis=0)
        loops.append(
            BoundaryLoop(
                points=points,
                vertex_indices=np.asarray(ordered, dtype=np.int64),
                center=center,
                normal=_polygon_normal(points),
                z_level=float(center[2]),
            )
        )

    if len(loops) < 2:
        raise ValueError(f"Expected at least two boundary loops, found {len(loops)}.")
    loops.sort(key=lambda loop: loop.center[2])
    return loops


def classify_top_bottom_boundaries(
        boundary_loops: Sequence[BoundaryLoop],
) -> Tuple[BoundaryLoop, BoundaryLoop]:
    """Classify bottom/top loops by center height."""
    if len(boundary_loops) < 2:
        raise ValueError("At least two boundary loops are required.")
    ordered = sorted(boundary_loops, key=lambda loop: loop.center[2])
    bottom = ordered[0]
    top = ordered[-1]
    bottom.name = "lowerBoundary"
    top.name = "upperBoundary"
    if np.dot(top.normal, np.array([0.0, 0.0, 1.0])) < 0:
        top.normal = -top.normal
    if np.dot(bottom.normal, np.array([0.0, 0.0, -1.0])) < 0:
        bottom.normal = -bottom.normal
    return top, bottom


def compute_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, axes and eigenvalues."""
    points = _as_array(points)
    if len(points) < 3:
        raise ValueError("PCA requires at least three points.")
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, axes = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order].T
    for index in range(axes.shape[0]):
        if axes[index, np.argmax(np.abs(axes[index]))] < 0:
            axes[index] *= -1.0
    return center, axes, values


def _upper_boundary_pca_second_point(
        top_boundary: BoundaryLoop, pca_second_dir: np.ndarray
) -> np.ndarray:
    points = top_boundary.points
    direction = _unit(pca_second_dir, "PCA second direction")
    projection = (points - top_boundary.center) @ direction
    return points[int(np.argmax(projection))]


def _upper_boundary_contact_point(top_boundary: BoundaryLoop) -> np.ndarray:
    """Find the upper-boundary contact point used by the reference plane."""
    points = _as_array(top_boundary.points)
    if len(points) < 3:
        raise ValueError("Upper boundary contact point requires at least three points.")

    center = top_boundary.center
    normal = _unit(top_boundary.normal, "upper boundary normal")
    u_axis, v_axis = _plane_basis(normal)
    points_2d = np.column_stack(
        ((points - center) @ u_axis, (points - center) @ v_axis)
    )

    def distance_to_polygon(point: np.ndarray) -> Tuple[float, int]:
        minimum_distance = float("inf")
        minimum_index = 0
        for index in range(len(points_2d)):
            next_index = (index + 1) % len(points_2d)
            start = points_2d[index]
            edge = points_2d[next_index] - start
            denominator = float(np.dot(edge, edge))
            ratio = 0.0 if denominator < EPS else float(
                np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0)
            )
            distance = float(np.linalg.norm(point - (start + ratio * edge)))
            if distance < minimum_distance:
                minimum_distance = distance
                minimum_index = index
        return minimum_distance, minimum_index

    current = np.zeros(2)
    current_radius, best_contact_index = distance_to_polygon(current)
    scale = max(float(np.linalg.norm(np.ptp(points_2d, axis=0))), 1.0)
    step_size = 0.1 if scale > 10.0 else 0.01 * scale
    for _ in range(100):
        improved = False
        for angle in np.arange(0.0, 2.0 * math.pi, math.pi / 4.0):
            candidate = current + step_size * np.array(
                [math.cos(angle), math.sin(angle)]
            )
            radius, contact_index = distance_to_polygon(candidate)
            if radius > current_radius:
                current = candidate
                current_radius = radius
                best_contact_index = contact_index
                improved = True
                break
        if not improved:
            break

    next_index = (best_contact_index + 1) % len(points)
    start_2d = points_2d[best_contact_index]
    edge_2d = points_2d[next_index] - start_2d
    denominator = float(np.dot(edge_2d, edge_2d))
    ratio = 0.0 if denominator < EPS else float(
        np.clip(np.dot(current - start_2d, edge_2d) / denominator, 0.0, 1.0)
    )
    return points[best_contact_index] + ratio * (
            points[next_index] - points[best_contact_index]
    )


def build_reference_plane(
        top_center: np.ndarray,
        bottom_center: np.ndarray,
        pca_second_point: np.ndarray,
) -> Plane:
    return _plane_from_three_points(
        np.asarray(bottom_center),
        np.asarray(top_center),
        np.asarray(pca_second_point),
    )


def _triangle_plane_segment(
        triangle_points: np.ndarray,
        plane: Plane,
        boundary_vertex_mask: Optional[Iterable[bool]] = None,
        eps: float = 1.0e-9,
) -> Optional[np.ndarray]:
    del boundary_vertex_mask
    distances = triangle_points @ plane.normal + plane.d
    distances[np.abs(distances) < eps] = 0.0
    points = []
    for index in range(3):
        next_index = (index + 1) % 3
        distance = distances[index]
        next_distance = distances[next_index]
        point = triangle_points[index]
        next_point = triangle_points[next_index]
        if distance == 0.0:
            points.append(point)
        if distance * next_distance < -eps:
            ratio = abs(distance) / (abs(distance) + abs(next_distance))
            points.append(point + ratio * (next_point - point))
        elif distance == 0.0 and next_distance == 0.0:
            points.extend([point, next_point])

    if len(points) < 2:
        return None
    points = _deduplicate_points(np.asarray(points), tol=1.0e-10)
    if len(points) < 2:
        return None
    if len(points) > 2:
        maximum_pair = (0, 1)
        maximum_distance = -1.0
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                distance = float(np.linalg.norm(points[first] - points[second]))
                if distance > maximum_distance:
                    maximum_distance = distance
                    maximum_pair = (first, second)
        points = points[list(maximum_pair)]
    if np.linalg.norm(points[0] - points[1]) < eps:
        return None
    return points


def _segments_to_curves(
        segments: Sequence[np.ndarray], tol: float
) -> List[np.ndarray]:
    if not segments:
        return []

    nodes: List[np.ndarray] = []
    key_to_index: dict[Tuple[int, int, int], int] = {}
    adjacency: dict[int, List[int]] = {}

    def node_index(point: np.ndarray) -> int:
        key = tuple(np.round(point / tol).astype(np.int64).tolist())
        if key not in key_to_index:
            key_to_index[key] = len(nodes)
            nodes.append(np.asarray(point, dtype=float))
        return key_to_index[key]

    for segment in segments:
        first = node_index(segment[0])
        second = node_index(segment[1])
        if first == second:
            continue
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)

    visited_edges: set[Tuple[int, int]] = set()

    def edge_key(first: int, second: int) -> Tuple[int, int]:
        return (first, second) if first < second else (second, first)

    curves: List[np.ndarray] = []
    starts = sorted(adjacency, key=lambda index: (len(adjacency[index]) != 1, index))
    for start in starts:
        while True:
            unused = [
                neighbor
                for neighbor in adjacency[start]
                if edge_key(start, neighbor) not in visited_edges
            ]
            if not unused:
                break
            ordered = [start]
            previous = -1
            current = start
            while True:
                choices = [
                    neighbor
                    for neighbor in adjacency[current]
                    if neighbor != previous
                       and edge_key(current, neighbor) not in visited_edges
                ]
                if not choices:
                    choices = [
                        neighbor
                        for neighbor in adjacency[current]
                        if edge_key(current, neighbor) not in visited_edges
                    ]
                if not choices:
                    break
                next_node = choices[0]
                visited_edges.add(edge_key(current, next_node))
                if next_node == ordered[0]:
                    ordered.append(next_node)
                    break
                ordered.append(next_node)
                previous, current = current, next_node
                if len(ordered) > len(nodes) + len(segments) + 2:
                    warnings.warn(
                        "Intersection curve tracing exceeded expected graph size."
                    )
                    break
            if len(ordered) >= 2:
                curves.append(np.vstack([nodes[index] for index in ordered]))

    curves.sort(
        key=lambda curve: _arc_lengths(curve)[-1] if len(curve) > 1 else 0.0,
        reverse=True,
    )
    return curves


def intersect_mesh_with_plane(
        mesh: trimesh.Trimesh, plane: Plane, tol: Optional[float] = None
) -> List[np.ndarray]:
    """Intersect all mesh triangles with a plane and stitch segments."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scale = max(float(np.linalg.norm(mesh.extents)), 1.0)
    eps = max(1.0e-10 * scale, 1.0e-9)
    if tol is None:
        tol = max(1.0e-7 * scale, 1.0e-8)
    segments = []
    for face in faces:
        segment = _triangle_plane_segment(vertices[face], plane, eps=eps)
        if segment is not None:
            segments.append(segment)
    return _segments_to_curves(segments, tol=tol)


def _select_mesh_file() -> str:
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select mesh for OptCuts",
        filetypes=[
            ("Mesh files", "*.stl *.obj *.ply *.off"),
            ("OBJ files", "*.obj"),
            ("STL files", "*.stl"),
            ("PLY files", "*.ply"),
            ("OFF files", "*.off"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [item for item in mesh.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("The selected file contains no triangular mesh.")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("The selected file contains no triangular mesh.")
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError("OptCuts requires triangular faces.")
    return mesh


def _prepare_official_input(mesh: trimesh.Trimesh) -> Path:
    input_directory = OPTCUTS_ROOT / "python_input"
    input_directory.mkdir(parents=True, exist_ok=True)
    path = input_directory / f"optcuts_{uuid.uuid4().hex}.obj"
    mesh.export(path, file_type="obj")
    return path


def dijkstra_path(adj: dict[int, list[int]], start: int, end: int, vertices: np.ndarray) -> list[int]:
    """Pure Python Dijkstra shortest path algorithm."""
    queue = [(0.0, start, [start])]
    min_costs = {start: 0.0}
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == end:
            return path
        if cost > min_costs.get(current, float('inf')):
            continue
        for neighbor in adj.get(current, []):
            edge_weight = float(np.linalg.norm(vertices[current] - vertices[neighbor]))
            new_cost = cost + edge_weight
            if new_cost < min_costs.get(neighbor, float('inf')):
                min_costs[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))
    return []


def trace_straight_seam(
        plane_edges: set[tuple[int, int]],
        new_bottom_vertices: set[int],
        new_top_vertices: set[int],
        vertices: np.ndarray,
        selected_start: np.ndarray,
        selected_end: np.ndarray
) -> list[int]:
    """Build an adjacency list from coplanar edges and trace a simple straight seam path."""
    adj: dict[int, list[int]] = {}
    for u, v in plane_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    start_candidates = list(new_bottom_vertices)
    if not start_candidates:
        raise ValueError("No bottom boundary vertices found to start tracing.")
    start_idx = min(start_candidates, key=lambda idx: np.linalg.norm(vertices[idx] - selected_start))

    end_candidates = list(new_top_vertices)
    if not end_candidates:
        raise ValueError("No top boundary vertices found to end tracing.")
    end_idx = min(end_candidates, key=lambda idx: np.linalg.norm(vertices[idx] - selected_end))

    path = dijkstra_path(adj, start_idx, end_idx, vertices)
    return path


def _precut_two_boundaries(
        mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, PrecutSeam | None]:
    """Cut a two-boundary mesh along loft's shorter three-point-plane curve by remeshing intersected triangles."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = ordered_boundary_loops(faces)
    if len(loops) != 2:
        return mesh, None

    boundary_loops = extract_boundary_loops(mesh)
    top, bottom = classify_top_bottom_boundaries(boundary_loops)
    _, pca_axes, _ = compute_pca(top.points)
    pca_second_direction = pca_axes[1]
    try:
        third_point = _upper_boundary_contact_point(top)
    except Exception:
        third_point = _upper_boundary_pca_second_point(top, pca_second_direction)
    plane = build_reference_plane(top.center, bottom.center, third_point)

    curves = [
        np.asarray(curve, dtype=float)
        for curve in intersect_mesh_with_plane(mesh, plane)
        if len(curve) >= 2 and _polyline_length(curve) > EPS
    ]
    if len(curves) < 2:
        raise ValueError("The loft three-point plane produced fewer than two usable intersections.")

    main_curves = sorted(curves, key=_polyline_length, reverse=True)[:2]
    curve_lengths = [_polyline_length(curve) for curve in main_curves]
    selected_index = int(np.argmin(curve_lengths))
    selected = main_curves[selected_index]

    vertices = np.asarray(mesh.vertices, dtype=float)
    bottom_ids = np.asarray(bottom.vertex_indices, dtype=np.int64)
    top_ids = np.asarray(top.vertex_indices, dtype=np.int64)

    forward_cost = (
            np.min(np.linalg.norm(vertices[bottom_ids] - selected[0], axis=1))
            + np.min(np.linalg.norm(vertices[top_ids] - selected[-1], axis=1))
    )
    reverse_cost = (
            np.min(np.linalg.norm(vertices[bottom_ids] - selected[-1], axis=1))
            + np.min(np.linalg.norm(vertices[top_ids] - selected[0], axis=1))
    )
    if reverse_cost < forward_cost:
        selected = selected[::-1].copy()

    # Remeshing Process
    edges = mesh_edges_from_faces(faces)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    typical_edge = float(np.median(edge_lengths)) if len(edge_lengths) > 0 else 1.0

    dense_curve = _dense_polyline_samples(selected, 0.2 * typical_edge)
    from scipy.spatial import cKDTree
    tree = cKDTree(dense_curve)
    tri_centers = vertices[faces].mean(axis=1)
    dists, _ = tree.query(tri_centers)

    new_vertices = list(vertices)
    new_faces = []
    edge_cut_cache = {}
    plane_edges = set()

    bottom_edges_set = {(min(a, b), max(a, b)) for a, b in zip(bottom_ids, np.roll(bottom_ids, -1))}
    top_edges_set = {(min(a, b), max(a, b)) for a, b in zip(top_ids, np.roll(top_ids, -1))}

    new_bottom_vertices = set(bottom_ids)
    new_top_vertices = set(top_ids)

    s_all = vertices @ plane.normal + plane.d
    s_all[np.abs(s_all) < 1.0e-9] = 0.0

    for f_idx, face in enumerate(faces):
        dist = dists[f_idx]
        s = s_all[face]
        is_near = dist < 1.5 * typical_edge

        if not is_near:
            new_faces.append(face.tolist())
            for idx_local in range(3):
                u, v = face[idx_local], face[(idx_local + 1) % 3]
                if s[idx_local] == 0.0 and s[(idx_local + 1) % 3] == 0.0:
                    plane_edges.add((min(u, v), max(u, v)))
            continue

        zero_mask = (s == 0.0)
        num_zeros = np.sum(zero_mask)

        if num_zeros == 3:
            new_faces.append(face.tolist())
            plane_edges.add((min(face[0], face[1]), max(face[0], face[1])))
            plane_edges.add((min(face[1], face[2]), max(face[1], face[2])))
            plane_edges.add((min(face[2], face[0]), max(face[2], face[0])))
            continue
        elif num_zeros == 2:
            new_faces.append(face.tolist())
            idx0 = np.where(zero_mask)[0]
            u, v = face[idx0[0]], face[idx0[1]]
            plane_edges.add((min(u, v), max(u, v)))
            continue
        elif num_zeros == 1:
            v_zero_local = np.where(zero_mask)[0][0]
            v_other_local = [(v_zero_local + 1) % 3, (v_zero_local + 2) % 3]
            s_other = s[v_other_local]
            if s_other[0] * s_other[1] < -1e-9:
                u0 = face[v_other_local[0]]
                u1 = face[v_other_local[1]]
                u2 = face[v_zero_local]
                su0 = s[v_other_local[0]]
                su1 = s[v_other_local[1]]

                edge_key = (min(u0, u1), max(u0, u1))
                if edge_key in edge_cut_cache:
                    p_idx = edge_cut_cache[edge_key]
                else:
                    t = abs(su0) / (abs(su0) + abs(su1))
                    p_xyz = vertices[u0] + t * (vertices[u1] - vertices[u0])
                    p_idx = len(new_vertices)
                    new_vertices.append(p_xyz)
                    edge_cut_cache[edge_key] = p_idx
                    if edge_key in bottom_edges_set:
                        new_bottom_vertices.add(p_idx)
                    if edge_key in top_edges_set:
                        new_top_vertices.add(p_idx)

                new_faces.append([u0, p_idx, u2])
                new_faces.append([p_idx, u1, u2])
                plane_edges.add((min(p_idx, u2), max(p_idx, u2)))
            else:
                new_faces.append(face.tolist())
            continue
        else:
            signs = np.sign(s)
            if signs[0] == signs[1] and signs[1] == signs[2]:
                new_faces.append(face.tolist())
                continue

            if signs[0] != signs[1] and signs[0] != signs[2]:
                v_single_local = 0
            elif signs[1] != signs[0] and signs[1] != signs[2]:
                v_single_local = 1
            else:
                v_single_local = 2

            u0 = face[v_single_local]
            u1 = face[(v_single_local + 1) % 3]
            u2 = face[(v_single_local + 2) % 3]

            su0 = s[v_single_local]
            su1 = s[(v_single_local + 1) % 3]
            su2 = s[(v_single_local + 2) % 3]

            edge01 = (min(u0, u1), max(u0, u1))
            if edge01 in edge_cut_cache:
                p1_idx = edge_cut_cache[edge01]
            else:
                t1 = abs(su0) / (abs(su0) + abs(su1))
                p1_xyz = vertices[u0] + t1 * (vertices[u1] - vertices[u0])
                p1_idx = len(new_vertices)
                new_vertices.append(p1_xyz)
                edge_cut_cache[edge01] = p1_idx
                if edge01 in bottom_edges_set:
                    new_bottom_vertices.add(p1_idx)
                if edge01 in top_edges_set:
                    new_top_vertices.add(p1_idx)

            edge02 = (min(u0, u2), max(u0, u2))
            if edge02 in edge_cut_cache:
                p2_idx = edge_cut_cache[edge02]
            else:
                t2 = abs(su0) / (abs(su0) + abs(su2))
                p2_xyz = vertices[u0] + t2 * (vertices[u2] - vertices[u0])
                p2_idx = len(new_vertices)
                new_vertices.append(p2_xyz)
                edge_cut_cache[edge02] = p2_idx
                if edge02 in bottom_edges_set:
                    new_bottom_vertices.add(p2_idx)
                if edge02 in top_edges_set:
                    new_top_vertices.add(p2_idx)

            new_faces.append([u0, p1_idx, p2_idx])
            new_faces.append([p1_idx, u1, u2])
            new_faces.append([p1_idx, u2, p2_idx])
            plane_edges.add((min(p1_idx, p2_idx), max(p1_idx, p2_idx)))

    new_vertices_arr = np.array(new_vertices, dtype=float)
    seam_path = trace_straight_seam(
        plane_edges, new_bottom_vertices, new_top_vertices,
        new_vertices_arr, selected[0], selected[-1]
    )

    remeshed_mesh = trimesh.Trimesh(
        vertices=new_vertices_arr,
        faces=np.array(new_faces, dtype=np.int64),
        process=False
    )

    original_vertex_count = len(remeshed_mesh.vertices)
    cut_mesh, _ = _cut_along_path(remeshed_mesh, np.array(seam_path, dtype=np.int64))

    seam_data = PrecutSeam(
        original_vertex_ids=np.asarray(seam_path, dtype=np.int64),
        cut_vertex_pairs=np.column_stack([
            seam_path,
            np.arange(original_vertex_count, original_vertex_count + len(seam_path), dtype=np.int64),
        ]),
        xyz=remeshed_mesh.vertices[seam_path].copy(),
    )

    after = len(ordered_boundary_loops(np.asarray(cut_mesh.faces, dtype=np.int64)))
    if after != 1:
        raise ValueError(
            "The loft three-point-plane seam did not produce a disk-like "
            f"OptCuts input (expected 1 boundary, found {after})."
        )

    print(
        "Two-boundary mesh: loft three-point plane generated two main "
        f"intersections ({curve_lengths[0]:.6g}, {curve_lengths[1]:.6g}); "
        f"selected the shorter one ({curve_lengths[selected_index]:.6g}). "
        f"Remeshed straight seam: {len(seam_path)} vertices, length "
        f"{_polyline_length(remeshed_mesh.vertices[seam_path]):.6g}."
    )
    return cut_mesh, seam_data


def _save_precut_seam(input_obj: Path, seam: PrecutSeam | None) -> Path | None:
    """Persist the main seam correspondence before OptCuts can add more cuts."""
    if seam is None:
        return None
    path = input_obj.with_suffix(".seam.json")
    payload = {
        "description": "Each cut_vertex_pairs row contains two OptCuts input vertex IDs for one physical seam point.",
        "index_base": 0,
        "original_vertex_ids": seam.original_vertex_ids.tolist(),
        "cut_vertex_pairs": seam.cut_vertex_pairs.tolist(),
        "xyz": seam.xyz.tolist(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved pre-cut seam correspondence: {path}")
    return path


def _polyline_length(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(values[1:] - values[:-1], axis=1)))


def _dense_polyline_samples(points: np.ndarray, step: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    samples = [values[0]]
    for start, stop in zip(values[:-1], values[1:]):
        length = float(np.linalg.norm(stop - start))
        count = max(1, int(np.ceil(length / step)))
        samples.extend(start + (stop - start) * ratio for ratio in np.linspace(0.0, 1.0, count + 1)[1:])
    return np.asarray(samples, dtype=float)


def _run_official_optcuts(input_obj: Path) -> Path:
    if not OPTCUTS_EXE.exists():
        raise FileNotFoundError(f"Official OptCuts executable was not found: {OPTCUTS_EXE}")
    output_root = OPTCUTS_ROOT / "output"
    output_root.mkdir(exist_ok=True)
    environment = os.environ.copy()
    dll_directory = OPTCUTS_ROOT / "build" / "stb_image" / "Release"
    environment["PATH"] = str(dll_directory) + os.pathsep + environment.get("PATH", "")
    command = [
        str(OPTCUTS_EXE),
        "10",
        input_obj.relative_to(OPTCUTS_ROOT).as_posix(),
        "0.999",
        "1",
        "0",
        "4.1",
        "1",
        "0",
        "python",
    ]
    completed = subprocess.run(command, cwd=OPTCUTS_ROOT, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Official OptCuts exited with code {completed.returncode}.")

    expected_prefix = input_obj.stem + "_"
    candidates = [
        result
        for result in output_root.glob("**/finalResult_mesh.obj")
        if result.parent.name.startswith(expected_prefix)
    ]
    if not candidates:
        raise FileNotFoundError(
            "This OptCuts run did not produce finalResult_mesh.obj; "
            "the run may have been stopped before completion."
        )
    result = max(candidates, key=lambda item: item.stat().st_mtime)
    print(f"Current OptCuts result: {result}")
    return result


def _read_obj_uv_data(path: Path) -> ObjUVData:
    vertices: list[list[float]] = []
    texture_coordinates: list[list[float]] = []
    vertex_faces: list[list[int]] = []
    texture_faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("vt "):
                values = line.split()
                texture_coordinates.append([float(values[1]), float(values[2])])
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    continue
                xyz_face: list[int] = []
                uv_face: list[int] = []
                for corner in corners:
                    fields = corner.split("/")
                    if len(fields) < 2 or not fields[0] or not fields[1]:
                        raise ValueError("OptCuts result contains an OBJ corner without UV data.")
                    xyz_face.append(int(fields[0]) - 1)
                    uv_face.append(int(fields[1]) - 1)
                vertex_faces.append(xyz_face)
                texture_faces.append(uv_face)
    data = ObjUVData(
        xyz=np.asarray(vertices, dtype=float),
        xyz_faces=np.asarray(vertex_faces, dtype=np.int64),
        uv=np.asarray(texture_coordinates, dtype=float),
        uv_faces=np.asarray(texture_faces, dtype=np.int64),
    )
    if not len(data.xyz) or not len(data.uv) or not len(data.xyz_faces):
        raise ValueError("Cannot read a complete XYZ/UV triangle mesh from the OptCuts result.")
    if len(data.xyz_faces) != len(data.uv_faces):
        raise ValueError("OBJ geometric and UV face counts differ.")
    return data


def _resolve_main_seam_uv_segments(data: ObjUVData, seam: PrecutSeam) -> list[SeamUVSegment]:
    edge_to_uv: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for xyz_face, uv_face in zip(data.xyz_faces, data.uv_faces):
        for local in range(3):
            vertex0 = int(xyz_face[local])
            vertex1 = int(xyz_face[(local + 1) % 3])
            uv0 = int(uv_face[local])
            uv1 = int(uv_face[(local + 1) % 3])
            key = (min(vertex0, vertex1), max(vertex0, vertex1))
            canonical = (uv0, uv1) if vertex0 < vertex1 else (uv1, uv0)
            edge_to_uv.setdefault(key, []).append(canonical)

    def uv_edge(vertex0: int, vertex1: int) -> tuple[np.ndarray, np.ndarray]:
        key = (min(vertex0, vertex1), max(vertex0, vertex1))
        candidates = edge_to_uv.get(key, [])
        if not candidates:
            raise ValueError(f"Saved main seam edge {key} is absent from the OptCuts result.")
        uv_pair = max(
            candidates,
            key=lambda item: float(np.linalg.norm(data.uv[item[1]] - data.uv[item[0]])),
        )
        first, second = uv_pair if vertex0 < vertex1 else uv_pair[::-1]
        return data.uv[first].copy(), data.uv[second].copy()

    segments: list[SeamUVSegment] = []
    pairs = np.asarray(seam.cut_vertex_pairs, dtype=np.int64)
    if len(pairs) < 2:
        raise ValueError("At least two saved seam point pairs are required for trajectory planning.")
    for index in range(len(pairs) - 1):
        a0, b0 = map(int, pairs[index])
        a1, b1 = map(int, pairs[index + 1])
        if max(a0, a1, b0, b1) >= len(data.xyz):
            raise ValueError("OptCuts changed the saved geometric vertex numbering.")
        uv_a0, uv_a1 = uv_edge(a0, a1)
        uv_b0, uv_b1 = uv_edge(b0, b1)
        segments.append(SeamUVSegment(index, uv_a0, uv_a1, uv_b0, uv_b1))
    return segments


def _planning_frame(seam_segments: Sequence[SeamUVSegment]) -> PlanningFrame:
    samples: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for segment in seam_segments:
        difference0 = segment.uv_b0 - segment.uv_a0
        difference_step = (
                (segment.uv_b1 - segment.uv_b0)
                - (segment.uv_a1 - segment.uv_a0)
        )
        denominator = float(np.dot(difference_step, difference_step))
        ratio = 0.0 if denominator <= EPS else float(np.clip(
            -np.dot(difference0, difference_step) / denominator, 0.0, 1.0
        ))
        uv_a = segment.uv_a0 + ratio * (segment.uv_a1 - segment.uv_a0)
        uv_b = segment.uv_b0 + ratio * (segment.uv_b1 - segment.uv_b0)
        samples.append((
            float(np.linalg.norm(uv_b - uv_a)),
            segment.physical_index,
            uv_a,
            uv_b,
        ))
    _, waist_index, left, right = min(samples, key=lambda item: item[0])
    reference = right - left
    width = float(np.linalg.norm(reference))
    if width <= EPS:
        raise ValueError("The two UV copies of the narrowest seam point coincide.")
    x_axis = reference / width
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=float)
    first, last = seam_segments[0], seam_segments[-1]
    lower_midpoint = 0.5 * (first.uv_a0 + first.uv_b0)
    upper_midpoint = 0.5 * (last.uv_a1 + last.uv_b1)
    if float(np.dot(upper_midpoint - lower_midpoint, y_axis)) < 0.0:
        y_axis *= -1.0
    return PlanningFrame(left.copy(), x_axis, y_axis, width, waist_index)


def _to_local(points: np.ndarray, frame: PlanningFrame) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    delta = values - frame.origin
    return np.column_stack([delta @ frame.x_axis, delta @ frame.y_axis])


def _from_local(points: np.ndarray, frame: PlanningFrame) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    return (frame.origin + values[:, :1] * frame.x_axis
            + values[:, 1:2] * frame.y_axis)


def _line_domain_intervals(
        local_uv: np.ndarray,
        uv_faces: np.ndarray,
        x_value: float,
        tolerance: float,
) -> list[tuple[float, float]]:
    triangles = np.asarray(local_uv, dtype=float)[np.asarray(uv_faces, dtype=np.int64)]
    selected = (
            (np.min(triangles[:, :, 0], axis=1) <= x_value + tolerance)
            & (np.max(triangles[:, :, 0], axis=1) >= x_value - tolerance)
    )
    triangles = triangles[selected]
    raw_intervals: list[tuple[float, float]] = []
    for triangle in triangles:
        intersections: list[float] = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            dx = float(b[0] - a[0])
            if abs(dx) <= tolerance:
                if abs(x_value - float(a[0])) <= tolerance:
                    intersections.extend((float(a[1]), float(b[1])))
                continue
            ratio = (x_value - float(a[0])) / dx
            if -tolerance <= ratio <= 1.0 + tolerance:
                intersections.append(float(a[1] + ratio * (b[1] - a[1])))
        if len(intersections) >= 2:
            low, high = min(intersections), max(intersections)
            if high - low > tolerance:
                raw_intervals.append((low, high))
    if not raw_intervals:
        return []
    raw_intervals.sort()
    merged = [raw_intervals[0]]
    for low, high in raw_intervals[1:]:
        previous_low, previous_high = merged[-1]
        if low <= previous_high + 10.0 * tolerance:
            merged[-1] = (previous_low, max(previous_high, high))
        else:
            merged.append((low, high))
    return merged


def _region_interval(
        local_uv: np.ndarray,
        uv_faces: np.ndarray,
        x_value: float,
        region: str,
        tolerance: float,
) -> tuple[float, float] | None:
    intervals = _line_domain_intervals(local_uv, uv_faces, x_value, tolerance)
    if not intervals:
        return None
    if region == "full":
        candidates = intervals
    elif region == "lower":
        candidates = [item for item in intervals if 0.5 * (item[0] + item[1]) <= tolerance]
    elif region == "upper":
        candidates = [item for item in intervals if 0.5 * (item[0] + item[1]) >= -tolerance]
    elif region != "full":
        raise ValueError(f"Unknown trajectory region: {region}")
    if not candidates:
        return None
    return min(item[0] for item in candidates), max(item[1] for item in candidates)


def optimize_spacing(nominal: float, dimensions: np.ndarray) -> Optimum:
    nominal = float(nominal)
    if nominal <= 0.0:
        raise ValueError("Nominal trajectory spacing must be greater than zero.")
    dimensions = np.asarray(dimensions, dtype=float)

    # 【校验还原】：新对称方案只需要 3 个维度（腰部、上部基准宽、下部基准宽）
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Spacing dimensions must contain [center, upper_ref, lower_ref] widths.")
    if dimensions[0] <= 0.0 or np.any(dimensions[1:] < 0.0):
        raise ValueError("Center width must be positive and reference widths cannot be negative.")

    candidates = np.linspace(0.9 * nominal, 1.1 * nominal, 500)
    center_ratio = dimensions[0] / candidates
    lower_odd = np.maximum(
        1, 2 * np.floor((center_ratio - 1.0) / 2.0).astype(int) + 1,
    )
    upper_odd = lower_odd + 2
    center_count = np.where(
        np.abs(dimensions[0] / lower_odd - candidates)
        <= np.abs(dimensions[0] / upper_odd - candidates),
        lower_odd,
        upper_odd,
    )

    # counts 矩阵回归为 3 列 (0: center, 1: upper_ref, 2: lower_ref)
    counts = np.zeros((len(candidates), 3), dtype=int)
    counts[:, 0] = center_count

    active = dimensions[None, :] > 0.75 * candidates[:, None]

    # 【工艺级安全控制】：对于外部两侧，改用 floor 向下取整，数学上确保实际间距 s >= d*，从而绝对防止最外侧越界
    counts[:, 1:] = np.where(
        active[:, 1:],
        np.maximum(
            1,
            np.floor(dimensions[None, 1:] / candidates[:, None]).astype(int),
        ),
        0,
    )

    actual = np.empty_like(counts, dtype=float)
    actual[:, 0] = dimensions[0] / counts[:, 0]
    for region in (1, 2):
        actual[:, region] = np.where(
            counts[:, region] > 0,
            dimensions[region] / np.maximum(counts[:, region], 1),
            0.5 * actual[:, 0] + dimensions[region],
        )

    # 3 维向量相对误差计算与多目标优化
    relative_error = (actual - candidates[:, None]) / candidates[:, None]
    objective_values = (
            np.var(relative_error, axis=1)
            + 0.5 * ((actual.mean(axis=1) - nominal) / nominal) ** 2
    )
    best = int(np.argmin(objective_values))

    return Optimum(
        spacing=float(candidates[best]),
        counts=counts[best].copy(),
        actual_spacings=actual[best].copy(),
        objective=float(objective_values[best]),
        candidates=candidates,
        objective_values=objective_values,
    )


def _balanced_center_positions(width: float, count: int) -> tuple[np.ndarray, float]:
    if count <= 0 or count % 2 == 0:
        raise ValueError("The central trajectory count must be a positive odd integer.")
    spacing = float(width) / int(count)
    positions = (np.arange(count, dtype=float) + 0.5) * spacing
    return positions, spacing


def _balanced_outer_positions(
        extent: float,
        count: int,
        waist_x: float,
        direction: float,
) -> tuple[np.ndarray, float]:
    extent = max(0.0, float(extent))
    if extent <= EPS or count <= 0:
        return np.empty(0, dtype=float), float("nan")
    actual = extent / count
    offsets = actual * (np.arange(count, dtype=float) + 0.5)
    return waist_x + direction * offsets, actual


class _UVToXYZMapper:
    def __init__(self, data: ObjUVData):
        from scipy.spatial import cKDTree

        self.data = data
        triangles = data.uv[data.uv_faces]
        self.triangle_minimum = np.min(triangles, axis=1)
        self.triangle_maximum = np.max(triangles, axis=1)
        self.triangle_centers = np.mean(triangles, axis=1)
        self.center_tree = cKDTree(self.triangle_centers)
        self.vertex_tree = cKDTree(data.uv)
        self.vertex_xyz = np.full((len(data.uv), 3), np.nan, dtype=float)
        uv_to_xyz: dict[int, list[np.ndarray]] = {}
        for xyz_face, uv_face in zip(data.xyz_faces, data.uv_faces):
            for xyz_index, uv_index in zip(xyz_face, uv_face):
                uv_to_xyz.setdefault(int(uv_index), []).append(data.xyz[int(xyz_index)])
        for uv_index, xyz_values in uv_to_xyz.items():
            self.vertex_xyz[uv_index] = np.mean(np.asarray(xyz_values, dtype=float), axis=0)
        self.uv_tolerance = 1.0e-8 * max(float(np.ptp(data.uv, axis=0).max()), 1.0)

    @staticmethod
    def _weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        a, b, c = triangle
        matrix = np.column_stack([b - a, c - a])
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= EPS:
            return np.array([np.nan, np.nan, np.nan])
        beta, gamma = np.linalg.solve(matrix, point - a)
        return np.array([1.0 - beta - gamma, beta, gamma], dtype=float)

    def map(
            self,
            points: np.ndarray,
            snap_tolerance: float | None = None,
            bridge_outside: bool = False,
    ) -> np.ndarray:
        uv_points = np.asarray(points, dtype=float)
        result = np.empty((len(uv_points), 3), dtype=float)
        uv_triangles = self.data.uv[self.data.uv_faces]
        face_count = len(uv_triangles)
        snap_limit = None if snap_tolerance is None else max(float(snap_tolerance), 0.0)
        for index, point in enumerate(uv_points):
            candidate_face = -1
            candidate_weights = None
            candidate_minimum = -np.inf
            query_count = min(32, face_count)
            while query_count <= min(2048, face_count):
                _, face_ids = self.center_tree.query(point, k=query_count)
                for current_face in np.atleast_1d(face_ids):
                    current_face = int(current_face)
                    if np.any(point < self.triangle_minimum[current_face] - self.uv_tolerance) or np.any(
                            point > self.triangle_maximum[current_face] + self.uv_tolerance
                    ):
                        continue
                    weights = self._weights(point, uv_triangles[current_face])
                    minimum = float(np.min(weights)) if np.isfinite(weights).all() else -np.inf
                    if minimum > candidate_minimum:
                        candidate_minimum = minimum
                        candidate_face = current_face
                        candidate_weights = weights
                    if minimum >= -1.0e-8:
                        break
                if candidate_minimum >= -1.0e-8 or query_count == face_count:
                    break
                next_count = min(2 * query_count, 2048, face_count)
                if next_count == query_count:
                    break
                query_count = next_count
            if candidate_face < 0 or candidate_minimum < -1.0e-5:
                if snap_limit is not None:
                    distance, uv_index = self.vertex_tree.query(point, k=1)
                    if float(distance) <= snap_limit:
                        snapped_xyz = self.vertex_xyz[int(uv_index)]
                        if np.isfinite(snapped_xyz).all():
                            result[index] = snapped_xyz
                            continue
                if bridge_outside:
                    result[index] = np.nan
                    continue
                raise ValueError(f"UV trajectory point lies outside the real domain: {point.tolist()}")
            face_id = candidate_face
            weights = np.asarray(candidate_weights, dtype=float)
            result[index] = weights @ self.data.xyz[self.data.xyz_faces[face_id]]
        if bridge_outside and not np.isfinite(result).all():
            valid = np.flatnonzero(np.isfinite(result).all(axis=1))
            if not len(valid):
                _, nearest_uv = self.vertex_tree.query(uv_points, k=1)
                result = self.vertex_xyz[np.asarray(nearest_uv, dtype=np.int64)].copy()
            else:
                sample_ids = np.arange(len(result), dtype=float)
                for coordinate in range(3):
                    result[:, coordinate] = np.interp(
                        sample_ids, valid.astype(float), result[valid, coordinate]
                    )
        return result


def _sample_scan_uv(
        x_value: float,
        interval: tuple[float, float],
        upward: bool,
        sample_step: float,
        frame: PlanningFrame,
) -> np.ndarray:
    start, stop = interval if upward else interval[::-1]
    count = max(2, int(np.ceil(abs(stop - start) / sample_step)) + 1)
    local = np.column_stack([
        np.full(count, x_value, dtype=float),
        np.linspace(start, stop, count),
    ])
    return _from_local(local, frame)


def _seam_projection_and_counterpart(
        point: np.ndarray,
        seam_segments: Sequence[SeamUVSegment],
        from_a: bool,
) -> tuple[np.ndarray, np.ndarray]:
    best_distance = np.inf
    best_source = None
    best_point = None
    for segment in seam_segments:
        if from_a:
            a, b = segment.uv_a0, segment.uv_a1
            target_a, target_b = segment.uv_b0, segment.uv_b1
        else:
            a, b = segment.uv_b0, segment.uv_b1
            target_a, target_b = segment.uv_a0, segment.uv_a1
        direction = b - a
        length2 = float(np.dot(direction, direction))
        ratio = 0.0 if length2 <= EPS else float(np.clip(np.dot(point - a, direction) / length2, 0.0, 1.0))
        projected = a + ratio * direction
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_source = projected
            best_point = target_a + ratio * (target_b - target_a)
    if best_source is None or best_point is None:
        raise ValueError("The saved seam contains no edge for periodic mapping.")
    return np.asarray(best_source, dtype=float), np.asarray(best_point, dtype=float)


def _seam_counterpart(
        point: np.ndarray,
        seam_segments: Sequence[SeamUVSegment],
        from_a: bool,
) -> np.ndarray:
    _, counterpart = _seam_projection_and_counterpart(
        point, seam_segments, from_a
    )
    return counterpart


def _overtravel_segment(
        previous: TrajectorySegment,
        following: TrajectorySegment,
        overtravel: float,
) -> TrajectorySegment:
    if len(previous.xyz) < 2 or len(following.xyz) < 2:
        raise ValueError("Surface scans need at least two points for 3D overtravel.")
    exit_direction = previous.xyz[-1] - previous.xyz[-2]
    entry_direction = following.xyz[1] - following.xyz[0]
    exit_direction /= max(float(np.linalg.norm(exit_direction)), EPS)
    entry_direction /= max(float(np.linalg.norm(entry_direction)), EPS)

    outside_direction = exit_direction - entry_direction
    outside_norm = float(np.linalg.norm(outside_direction))
    if outside_norm <= EPS:
        outside_direction = exit_direction
    else:
        outside_direction /= outside_norm
    xyz = np.vstack([
        previous.xyz[-1],
        previous.xyz[-1] + overtravel * outside_direction,
        following.xyz[0] + overtravel * outside_direction,
        following.xyz[0],
    ])
    note = "3D-only opening connector"

    return TrajectorySegment(
        kind="OVERTRAVEL",
        uv=np.full((len(xyz), 2), np.nan),
        xyz=xyz,
        spray_on=False,
        note="3D-only opening connector",
        region="TRANSITION"  # 【新增】：标记为过渡段
    )


def _trim_mapped_segment(
        segment: TrajectorySegment,
        distance: float,
        from_start: bool,
) -> TrajectorySegment:
    xyz = np.asarray(segment.xyz, dtype=float)
    uv = np.asarray(segment.uv, dtype=float)
    if from_start:
        reversed_segment = TrajectorySegment(
            segment.kind, segment.uv[::-1].copy(), segment.xyz[::-1].copy(),
            segment.spray_on, segment.note,
        )
        trimmed = _trim_mapped_segment(reversed_segment, distance, False)
        return TrajectorySegment(
            trimmed.kind, trimmed.uv[::-1].copy(), trimmed.xyz[::-1].copy(),
            trimmed.spray_on, trimmed.note,
        )
    if len(xyz) < 2:
        raise ValueError("A seam-adjacent scan needs at least two 3D points.")
    edge_lengths = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    cumulative = np.r_[0.0, np.cumsum(edge_lengths)]
    total = float(cumulative[-1])
    target = total - float(distance)
    if target <= EPS:
        raise ValueError("The requested 3D seam blend would consume an entire scan.")
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(xyz) - 2)
    length = float(edge_lengths[index])
    ratio = 0.0 if length <= EPS else (target - cumulative[index]) / length
    xyz_endpoint = xyz[index] + ratio * (xyz[index + 1] - xyz[index])
    if len(uv) == len(xyz):
        uv_endpoint = uv[index] + ratio * (uv[index + 1] - uv[index])
        trimmed_uv = np.vstack([uv[:index + 1], uv_endpoint])
    else:
        trimmed_uv = np.full((index + 2, 2), np.nan)
    return TrajectorySegment(
        segment.kind,
        trimmed_uv,
        np.vstack([xyz[:index + 1], xyz_endpoint]),
        segment.spray_on,
        segment.note,
    )


def _cubic_xyz_blend(
        start: np.ndarray,
        stop: np.ndarray,
        start_tangent: np.ndarray,
        stop_tangent: np.ndarray,
        nominal_radius: float,
        sample_step: float,
) -> np.ndarray:
    start_tangent = np.asarray(start_tangent, dtype=float)
    stop_tangent = np.asarray(stop_tangent, dtype=float)
    start_tangent /= max(float(np.linalg.norm(start_tangent)), EPS)
    stop_tangent /= max(float(np.linalg.norm(stop_tangent)), EPS)
    chord = float(np.linalg.norm(stop - start))
    if chord <= EPS:
        return np.vstack([start, stop])
    turn = float(np.arccos(np.clip(np.dot(start_tangent, stop_tangent), -1.0, 1.0)))
    circular_handle = 4.0 * nominal_radius / 3.0 * np.tan(max(turn, 1.0e-3) / 4.0)
    handle = min(max(circular_handle, 0.16 * chord), 0.48 * chord)
    control = np.vstack([
        start,
        start + handle * start_tangent,
        stop - handle * stop_tangent,
        stop,
    ])
    control_length = float(np.sum(np.linalg.norm(control[1:] - control[:-1], axis=1)))
    count = max(16, int(np.ceil(control_length / sample_step)) + 1)
    phase = np.linspace(0.0, np.pi, count)
    parameter = (0.5 - 0.5 * np.cos(phase))[:, None]
    xyz = ((1.0 - parameter) ** 3 * control[0]
           + 3.0 * (1.0 - parameter) ** 2 * parameter * control[1]
           + 3.0 * (1.0 - parameter) * parameter ** 2 * control[2]
           + parameter ** 3 * control[3])
    xyz[0] = start
    xyz[-1] = stop
    return xyz


def _smooth_seam_jumps_xyz(
        segments: Sequence[TrajectorySegment],
        nominal_radius: float,
        sample_step: float,
) -> list[TrajectorySegment]:
    result: list[TrajectorySegment] = []
    index = 0
    while index < len(segments):
        if (
                index + 2 < len(segments)
                and segments[index].kind == "SURFACE_SCAN"
                and segments[index + 1].kind == "SEAM_JUMP"
                and segments[index + 2].kind == "SURFACE_SCAN"
        ):
            incoming = segments[index]
            jump = segments[index + 1]
            outgoing = segments[index + 2]
            incoming_length = float(np.sum(np.linalg.norm(
                incoming.xyz[1:] - incoming.xyz[:-1], axis=1
            )))
            outgoing_length = float(np.sum(np.linalg.norm(
                outgoing.xyz[1:] - outgoing.xyz[:-1], axis=1
            )))
            lead = min(nominal_radius, 0.45 * incoming_length, 0.45 * outgoing_length)
            if Terror := lead <= EPS:
                result.extend([incoming, jump, outgoing])
                index += 3
                continue
            incoming_trimmed = _trim_mapped_segment(incoming, lead, from_start=False)
            outgoing_trimmed = _trim_mapped_segment(outgoing, lead, from_start=True)
            incoming_tangent = incoming_trimmed.xyz[-1] - incoming_trimmed.xyz[-2]
            outgoing_tangent = outgoing_trimmed.xyz[1] - outgoing_trimmed.xyz[0]
            blend_xyz = _cubic_xyz_blend(
                incoming_trimmed.xyz[-1], outgoing_trimmed.xyz[0],
                incoming_tangent, outgoing_tangent,
                nominal_radius, sample_step,
            )
            marker = TrajectorySegment(
                "SEAM_JUMP",
                jump.uv.copy(),
                np.empty((0, 3), dtype=float),
                False,
                jump.note + "; XYZ motion replaced by SEAM_BLEND_3D",
            )
            blend = TrajectorySegment(
                "SEAM_BLEND_3D",
                np.full((len(blend_xyz), 2), np.nan),
                blend_xyz,
                incoming.spray_on and outgoing.spray_on,
                (f"3D coordinated seam blend; nominal radius={nominal_radius:.12g}; "
                 f"lead distance={lead:.12g}"),
            )
            result.extend([incoming_trimmed, marker, blend, outgoing_trimmed])
            index += 3
            continue
        result.append(segments[index])
        index += 1
    return result


def _make_surface_scan(
        x_value: float,
        interval: tuple[float, float],
        upward: bool,
        sample_step: float,
        frame: PlanningFrame,
        mapper: _UVToXYZMapper,
        snap_tolerance: float | None = None,
        note: str = "",
        region: str = "",
) -> TrajectorySegment:
    uv = _sample_scan_uv(x_value, interval, upward, sample_step, frame)
    xyz = mapper.map(uv, snap_tolerance, bridge_outside=True)

    # 【新增】：生成 0/1 阶段数组（前半部分为 0，后半部分为 1）
    n_pts = len(xyz)
    phases = np.where(np.arange(n_pts) < n_pts // 2, 0, 1)

    return TrajectorySegment(
        "SURFACE_SCAN", uv, xyz, True, note, region,
        phases=phases  # 传入 phases
    )


def _make_scan_from_seam_destination(
        destination_uv: np.ndarray,
        region: str,
        upward: bool,
        sample_step: float,
        frame: PlanningFrame,
        local_uv: np.ndarray,
        uv_faces: np.ndarray,
        tolerance: float,
        mapper: _UVToXYZMapper,
        snap_tolerance: float | None = None,
        note: str = "",
) -> TrajectorySegment | None:
    dest_local = _to_local(np.asarray(destination_uv, dtype=float)[None, :], frame)[0]
    x_val = float(dest_local[0])
    dest_y = float(dest_local[1])

    all_intervals = _line_domain_intervals(local_uv, uv_faces, x_val, tolerance)
    if not all_intervals:
        return None
    interval = _region_interval(local_uv, uv_faces, x_val, region, tolerance)
    if interval is None:
        interval = (
            min(item[0] for item in all_intervals),
            max(item[1] for item in all_intervals),
        )

    low, high = interval
    if upward:
        low = float(np.clip(dest_y, low, high))
    else:
        high = float(np.clip(dest_y, low, high))

    if abs(high - low) < tolerance:
        return None

    uv = _sample_scan_uv(x_val, (low, high), upward, sample_step, frame)
    uv[0] = np.asarray(destination_uv, dtype=float).copy()
    return TrajectorySegment(
        "SURFACE_SCAN", uv,
        mapper.map(uv, snap_tolerance, bridge_outside=True), True, note,
    )


def _available_tracks(
        positions: Iterable[float],
        region: str,
        local_uv: np.ndarray,
        uv_faces: np.ndarray,
        tolerance: float,
) -> list[tuple[float, tuple[float, float]]]:
    tracks: list[tuple[float, tuple[float, float]]] = []
    for x_value in positions:
        interval = _region_interval(local_uv, uv_faces, float(x_value), region, tolerance)
        if interval is not None:
            tracks.append((float(x_value), interval))
    return tracks


def _available_center_tracks(
        positions: Iterable[float],
        local_uv: np.ndarray,
        uv_faces: np.ndarray,
        tolerance: float,
) -> tuple[list[tuple[float, tuple[float, float]]], int, int]:
    tracks: list[tuple[float, tuple[float, float]]] = []
    for x_value in positions:
        intervals = _line_domain_intervals(local_uv, uv_faces, float(x_value), tolerance)
        if not intervals:
            continue
        interval = (
            min(item[0] for item in intervals),
            max(item[1] for item in intervals),
        )
        tracks.append((float(x_value), interval))
    return tracks, 0, 0


def smooth_v_trajectory(
        xyz: np.ndarray,
        min_distance: float = 0.2,
        smooth_iterations: int = 2,
        smooth_weight: float = 0.25
) -> np.ndarray:
    """
    针对 V 字型/切缝折返轨迹的噪点清理与平滑函数

    :param xyz: 原始 3D 轨迹点阵 (N x 3)
    :param min_distance: 去重距离阈值 (mm)，小于此距离的点将被合并，解决图2粗线重叠问题
    :param smooth_iterations: 平滑迭代次数 (1-3次即可，不宜过多以免过度变形)
    :param smooth_weight: 平滑权重 (0.1 ~ 0.3)
    :return: 清理平滑后的 3D 轨迹点阵
    """
    if len(xyz) < 3:
        return xyz

    # ---------------------------------------------------------
    # 步骤 1: 3D 点去重 (解决尖角处密集噪点拧成粗线的问题)
    # ---------------------------------------------------------
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    # 保留首尾点，以及与前一点距离大于 min_distance 的点
    keep_mask = np.r_[True, dists > min_distance]
    cleaned_xyz = xyz[keep_mask]

    if len(cleaned_xyz) < 3:
        return cleaned_xyz

    # ---------------------------------------------------------
    # 步骤 2: 拉普拉斯滤波 (消除切缝附近的微小锯齿抖动)
    # ---------------------------------------------------------
    smoothed_xyz = cleaned_xyz.copy()
    for _ in range(smooth_iterations):
        # 保持轨迹起点和终点不变，只平滑中间点
        laplacian = 0.5 * (smoothed_xyz[:-2] + smoothed_xyz[2:]) - smoothed_xyz[1:-1]
        smoothed_xyz[1:-1] += smooth_weight * laplacian

    return smoothed_xyz


def extract_raw_v_trajectories(
        trajectory: List[TrajectorySegment],
) -> List[dict]:
    """
    第一阶段：提取 LOWER/UPPER 区域的原始 V 形轨迹特征及关键控制点。
    严格保证 P1(起点)、P3(终点) 作为固定端点，P2 为原始几何尖点。
    """
    v_segments = []
    for seg in trajectory:
        if seg.kind == "SURFACE_SCAN" and getattr(seg, "region", "") in ("LOWER", "UPPER"):
            xyz = np.asarray(seg.xyz, dtype=float)
            if len(xyz) < 3:
                v_segments.append({"type": "PASSTHROUGH", "seg": seg})
                continue

            P1 = xyz[0]
            P3 = xyz[-1]
            chord_vec = P3 - P1
            chord_len = np.linalg.norm(chord_vec)
            if chord_len < 1e-6:
                v_segments.append({"type": "PASSTHROUGH", "seg": seg})
                continue

            u_chord = chord_vec / chord_len
            vecs = xyz - P1
            proj_lengths = vecs @ u_chord
            proj_pts = P1 + np.outer(proj_lengths, u_chord)
            perp_dists = np.linalg.norm(xyz - proj_pts, axis=1)
            idx_tip = int(np.argmax(perp_dists))
            P2 = xyz[idx_tip]

            v_segments.append({
                "type": "V_SHAPE",
                "seg": seg,
                "xyz": xyz,
                "idx_tip": idx_tip,
                "P1": P1,
                "P2": P2,
                "P3": P3,
            })
        else:
            v_segments.append({"type": "PASSTHROUGH", "seg": seg})
    return v_segments


def fit_superellipse_v_trajectories(
        extracted_v_data: List[dict],
        mesh: trimesh.Trimesh,
        speed: float,
        a_max: float,
        j_max: float,
        target_spacing: float = 30.0,
        sample_step: float = 1.0,
) -> List[TrajectorySegment]:
    """
    全自由度高保真超椭圆拟合器：
    - 端点 P1, P3 严格绝对锁定；
    - 联合优化 [形态指数 n, 顶点高度 b, 轴向倾角 theta]，实现对原始轨迹的最大逼近保真度；
    - 采用点云几何正交投影距离作为损失函数，避免索引错位导致的轨迹凹瘪畸变。
    """
    effective_step = sample_step if sample_step > 1e-6 else 1.0
    fitted_segments: List[TrajectorySegment] = []

    for item in extracted_v_data:
        if item["type"] == "PASSTHROUGH":
            fitted_segments.append(item["seg"])
            continue

        seg = item["seg"]
        xyz_raw = np.asarray(item["xyz"], dtype=float)
        P1 = item["P1"]
        P3 = item["P3"]
        P2 = item["P2"]

        if len(xyz_raw) < 5:
            fitted_segments.append(seg)
            continue

        # ---------------------------------------------------------------------
        # 1. 建立 3D 几何参考基底 (端点 P1, P3 严格锁定)
        # ---------------------------------------------------------------------
        M = 0.5 * (P1 + P3)
        X_axis = P3 - P1
        span_2a = np.linalg.norm(X_axis)
        if span_2a < 1e-6:
            fitted_segments.append(seg)
            continue
        X_dir = X_axis / span_2a
        a = span_2a / 2.0

        # 初始高度矢量
        height_vec = P2 - M
        h_comp = height_vec - np.dot(height_vec, X_dir) * X_dir
        b_init = float(np.linalg.norm(h_comp))
        if b_init < 1e-6:
            fitted_segments.append(seg)
            continue
        Y_dir_base = h_comp / b_init
        Z_dir = np.cross(X_dir, Y_dir_base)
        Z_norm = np.linalg.norm(Z_dir)
        if Z_norm > 1e-6:
            Z_dir /= Z_norm
        else:
            Z_dir = np.array([0.0, 0.0, 1.0])

        # ---------------------------------------------------------------------
        # 2. 参数化超椭圆生成器：支持 n、高度 b、空间倾角 theta
        # ---------------------------------------------------------------------
        def eval_superellipse_3d(n_val, b_val, theta_val, num_pts=100):
            t = np.linspace(math.pi, 0.0, num_pts)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            exp_p = 2.0 / max(n_val, 0.05)

            x_loc = a * np.sign(cos_t) * (np.abs(cos_t) ** exp_p)
            y_loc = b_val * (np.abs(sin_t) ** exp_p)

            # 沿法向方向允许微小旋转倾斜 theta
            Y_dir_t = Y_dir_base * np.cos(theta_val) + Z_dir * np.sin(theta_val)
            curve_pts = M[None, :] + np.outer(x_loc, X_dir) + np.outer(y_loc, Y_dir_t)
            return curve_pts

        # ---------------------------------------------------------------------
        # 3. 高保真优化目标：最小化“原始点阵”与“生成超椭圆”之间的正交几何投影误差
        # ---------------------------------------------------------------------
        tree_raw = cKDTree(xyz_raw)

        def objective_loss(params):
            n_curr, b_curr, theta_curr = params
            # 生成稠密测试样条
            test_curve = eval_superellipse_3d(n_curr, b_curr, theta_curr, num_pts=80)

            # 双向几何距离均方差：
            # 1. 拟合线上的点到原始点的距离
            dists_c2r, _ = tree_raw.query(test_curve)
            # 2. 原始点到拟合线上的距离
            tree_test = cKDTree(test_curve)
            dists_r2c, _ = tree_test.query(xyz_raw)

            loss = np.mean(dists_c2r ** 2) + np.mean(dists_r2c ** 2)
            return loss

        # 自由度全开的搜索边界：
        # n: [0.1, 2.5] (全面涵盖内摆线、直边三角、勒洛三角、圆弧/椭圆)
        # b: [0.5 * b_init, 1.3 * b_init] (给高度充分自由度，不锁死)
        # theta: [-15°, +15°] (适应曲面法向微小扭转)
        bounds = [
            (0.10, 2.50),
            (0.50 * b_init, 1.30 * b_init),
            (-np.radians(15.0), np.radians(15.0))
        ]

        # 多初始点探索以防局部极小
        best_res = None
        best_loss = float("inf")
        for n_try in [0.5, 1.0, 1.5, 2.0]:
            init_x = [n_try, b_init, 0.0]
            res = minimize(objective_loss, init_x, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_loss:
                best_loss = res.fun
                best_res = res

        best_n, best_b, best_theta = best_res.x

        # ---------------------------------------------------------------------
        # 4. 生成最终高密度连续 3D 超椭圆，并贴附 STL 网格曲面
        # ---------------------------------------------------------------------
        # 根据弧长动态决定等距点数
        arc_est = math.pi * math.sqrt(0.5 * (a ** 2 + best_b ** 2))
        num_final = max(32, int(np.ceil(arc_est / effective_step)) + 1)

        final_curve = eval_superellipse_3d(best_n, best_b, best_theta, num_pts=num_final)

        # 刚性锁定首尾端点 (P1, P3 绝不移动)
        final_curve[0] = P1
        final_curve[-1] = P3

        # 投影吸附回 STL 三角网格表面
        proj_curve, _, _ = trimesh.proximity.closest_point(mesh, final_curve)
        proj_curve[0] = P1
        proj_curve[-1] = P3

        # 判定拟合形态
        if best_n < 0.85:
            shape_type = "三角内摆线"
        elif best_n <= 1.15:
            shape_type = "标准三角"
        elif best_n < 1.95:
            shape_type = "勒洛三角"
        else:
            shape_type = "标准椭圆"

        n_pts = len(proj_curve)
        phases = np.where(np.arange(n_pts) < n_pts // 2, 0, 1)

        geometry_meta = {
            "type": "SUPERELLIPSE",
            "a": float(a),
            "b": float(best_b),
            "n": float(best_n),
            "theta_deg": float(np.degrees(best_theta)),
            "P1": P1.tolist(),
            "P3": P3.tolist(),
        }

        fitted_segments.append(TrajectorySegment(
            kind=seg.kind,
            uv=np.full((n_pts, 2), np.nan),
            xyz=proj_curve,
            spray_on=seg.spray_on,
            note=seg.note.split(" [")[0] + f" [{shape_type} n={best_n:.2f}, b={best_b:.1f}]",
            region=seg.region,
            phases=phases,
            geometry_meta=geometry_meta
        ))

    return fitted_segments


def prune_v_sharp_apex(
        xyz: np.ndarray,
        cutoff_distance: float,
        mesh: trimesh.Trimesh | None = None
) -> np.ndarray:
    """
    鲁棒的 V 字折返截断：通过弦向距离精确找到折返尖点 P2，再向两臂回溯截断。
    """
    if len(xyz) < 8:
        return xyz

    # 1. 严格通过离首尾弦线最远的点找到几何折返尖点 P2
    P1, P3 = xyz[0], xyz[-1]
    chord = P3 - P1
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-6:
        return xyz
    u_chord = chord / chord_len
    vecs = xyz - P1
    perp_dists = np.linalg.norm(vecs - np.outer(vecs @ u_chord, u_chord), axis=1)
    tip_idx = int(np.argmax(perp_dists))

    if tip_idx <= 2 or tip_idx >= len(xyz) - 3:
        return xyz

    # 2. 拆分为进气臂与出气臂
    leg1 = xyz[:tip_idx + 1]  # 0 -> tip
    leg2 = xyz[tip_idx:]      # tip -> end

    # 3. 计算两臂欧氏距离矩阵
    diffs = np.linalg.norm(leg1[:, None, :] - leg2[None, :, :], axis=2)
    under_cutoff = diffs < cutoff_distance

    if not np.any(under_cutoff):
        return xyz

    i_indices, j_indices = np.where(under_cutoff)
    idx1 = max(0, int(np.min(i_indices)) - 1)
    idx2 = min(len(leg2) - 1, int(np.max(j_indices)) + 1)

    if idx1 >= len(leg1) or idx2 <= 0:
        return xyz

    # 4. 截断两臂并在截断中点插入平滑新顶点
    p1 = leg1[idx1]
    p2 = leg2[idx2]
    p_mid = 0.5 * (p1 + p2)

    new_xyz = np.vstack([
        leg1[:idx1 + 1],
        p_mid[None, :],
        leg2[idx2:]
    ])

    if mesh is not None:
        proj_mid, _, _ = trimesh.proximity.closest_point(mesh, p_mid[None, :])
        new_xyz[idx1 + 1] = proj_mid[0]

    return new_xyz


def optimize_trajectory_arcs_postprocess(
        trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        speed: float = 300.0,
        a_max: float = 1000.0,
        j_max: float = 5000.0,
        target_spacing: float = 30.0,
        sample_step: float = 1.0,
        **kwargs
) -> List[TrajectorySegment]:
    """
    统一入口：
    第一步：调用 prune_v_sharp_apex 彻底截断两臂间距 < d/2 的死区
    第二步：调用 extract_raw_v_trajectories 提取截断后的真实特征点
    第三步：调用 fit_superellipse_v_trajectories 执行全自由度超椭圆拟合
    """
    pruned_trajectory: List[TrajectorySegment] = []
    cutoff_d = 2.0 * float(target_spacing)  # d/2 截断阈值

    # 1. 执行物理截断
    for seg in trajectory:
        if seg.kind == "SURFACE_SCAN" and getattr(seg, "region", "") in ("LOWER", "UPPER"):
            xyz = np.asarray(seg.xyz, dtype=float)
            if len(xyz) >= 8:
                xyz_pruned = prune_v_sharp_apex(xyz, cutoff_distance=cutoff_d, mesh=mesh)
                n_p = len(xyz_pruned)
                phases_p = np.where(np.arange(n_p) < n_p // 2, 0, 1)

                pruned_trajectory.append(TrajectorySegment(
                    kind=seg.kind,
                    uv=seg.uv,
                    xyz=xyz_pruned,
                    spray_on=seg.spray_on,
                    note=seg.note,
                    region=seg.region,
                    phases=phases_p,
                    geometry_meta=seg.geometry_meta
                ))
            else:
                pruned_trajectory.append(seg)
        else:
            pruned_trajectory.append(seg)

    # 2. 提取截断后的特征控制点
    v_data = extract_raw_v_trajectories(pruned_trajectory)

    # 3. 真正执行高保真超椭圆拟合
    fitted_trajectory = fit_superellipse_v_trajectories(
        v_data,
        mesh=mesh,
        speed=speed,
        a_max=a_max,
        j_max=j_max,
        target_spacing=target_spacing,
        sample_step=sample_step,
    )
    return fitted_trajectory

#            CENTER 区域 双空白能量场优化器（最终版，纯 3D，不依赖 UV）
# =========================================================================

def _resample_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """按弧长均匀重采样折线（钉死首尾端点）。"""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2 or step <= 1e-9:
        return pts
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    total = float(cum[-1])
    if total <= 1e-9:
        return pts
    n = max(2, int(round(total / step)) + 1)
    s = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(s, cum, pts[:, k])
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _void_centers(
        tri_centers: np.ndarray,
        fixed_pts: np.ndarray,
        center_pts: np.ndarray,
        d: float,
        n_centers: int = 2,
) -> tuple:
    """自动定位 mesh 上最大的 n_centers 个空白区中心。
    与部件朝向、圆弧区域存在几个均无关。
    返回 (centers, max_void_depth)；max_void_depth 用于判断是否存在大空白（缺弧）。"""
    from scipy.spatial import cKDTree
    all_pts = np.vstack([fixed_pts, center_pts]) if len(fixed_pts) else center_pts
    dist, _ = cKDTree(all_pts).query(tri_centers)

    void_mask = dist > 0.6 * d
    void_pts = tri_centers[void_mask]
    void_w = dist[void_mask] - 0.6 * d
    if len(void_pts) == 0:
        return [], 0.0  # 【鲁棒性3】：无空白 -> 空列表 + 深度 0
    max_void_depth = float(void_w.max())

    order = np.argsort(void_w)[::-1]
    seeds = []
    for idx in order:
        c = void_pts[idx]
        if all(float(np.linalg.norm(c - s)) > 3.0 * d for s in seeds):
            seeds.append(c)
        if len(seeds) == n_centers:
            break

    centers = []
    for s in seeds:
        local = np.linalg.norm(void_pts - s, axis=1) <= 1.5 * d
        if np.any(local):
            w = np.maximum(void_w[local], 1e-6)
            centers.append((void_pts[local] * w[:, None]).sum(axis=0) / w.sum())
        else:
            centers.append(s)
    return centers, max_void_depth


def _trajectory_objective(
        tri_centers: np.ndarray,
        fixed_pts: np.ndarray,
        lines: Sequence[np.ndarray],
        target_spacing: float,
        coverage_weight: float,
        spacing_weight: float,
        boundary_weight: float,
        smooth_weight: float,
) -> tuple[float, dict[str, float]]:
    """Evaluate one fixed, dimensionless trajectory-quality objective.

    Every component is a mean squared loss, so changing the resampling point
    count does not by itself change the reported energy.
    """
    d = max(float(target_spacing), EPS)
    center_pts = np.vstack(lines)
    all_scan_pts = (
        np.vstack([fixed_pts, center_pts]) if len(fixed_pts) else center_pts
    )

    # Mesh samples farther than half the desired track spacing are uncovered.
    coverage_distance, _ = cKDTree(all_scan_pts).query(tri_centers)
    coverage_residual = np.maximum(coverage_distance / d - 0.5, 0.0)
    coverage_loss = float(np.mean(coverage_residual ** 2))

    # Adjacent center trajectories should remain one target spacing apart.
    spacing_samples: list[np.ndarray] = []
    for first, second in zip(lines[:-1], lines[1:]):
        first_to_second, _ = cKDTree(second).query(first)
        second_to_first, _ = cKDTree(first).query(second)
        spacing_samples.extend([first_to_second, second_to_first])
    if spacing_samples:
        spacing_distance = np.concatenate(spacing_samples)
        spacing_loss = float(np.mean((spacing_distance / d - 1.0) ** 2))
    else:
        spacing_loss = 0.0

    # CENTER paths must not enter the protected LOWER/UPPER arc zone.
    if len(fixed_pts):
        boundary_distance, _ = cKDTree(fixed_pts).query(center_pts)
        boundary_residual = np.maximum(1.15 - boundary_distance / d, 0.0)
        boundary_loss = float(np.mean(boundary_residual ** 2))
    else:
        boundary_loss = 0.0

    # Penalize discrete curvature; normalize by d to make it dimensionless.
    curvature_samples = [
        np.linalg.norm(line[:-2] - 2.0 * line[1:-1] + line[2:], axis=1) / d
        for line in lines if len(line) > 2
    ]
    smooth_loss = (
        float(np.mean(np.concatenate(curvature_samples) ** 2))
        if curvature_samples else 0.0
    )

    components = {
        "coverage": coverage_loss,
        "spacing": spacing_loss,
        "boundary": boundary_loss,
        "smooth": smooth_loss,
    }
    total = (
            coverage_weight * coverage_loss
            + spacing_weight * spacing_loss
            + boundary_weight * boundary_loss
            + smooth_weight * smooth_loss
    )
    return float(total), components


def _extract_arc_geometries(trajectory: List[TrajectorySegment]) -> list[dict]:
    """提取 LOWER/UPPER 区域圆弧的 2D 投影平面几何信息 (M, u, v, radius)"""
    arcs = []
    for seg in trajectory:
        if seg.kind == "SURFACE_SCAN" and getattr(seg, "region", "") in ("LOWER", "UPPER"):
            xyz = np.asarray(seg.xyz, dtype=float)
            if len(xyz) < 3:
                continue
            p1, p3 = xyz[0], xyz[-1]
            chord = p3 - p1
            chord_len = float(np.linalg.norm(chord))
            if chord_len < 1e-6:
                continue
            u = chord / chord_len
            m = 0.5 * (p1 + p3)

            # 寻找拱起峰值点 P2，确定正交方向 v (指向圆弧凸起方向)
            vecs = xyz - p1
            perps = vecs - np.outer(vecs @ u, u)
            p2 = xyz[int(np.argmax(np.linalg.norm(perps, axis=1)))]

            w = p2 - m
            v = w - np.dot(w, u) * u
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-6:
                continue
            v /= v_norm
            radius = chord_len / 2.0

            arcs.append({
                "midpoint": m,
                "u": u,
                "v": v,
                "radius": radius,
                "region": getattr(seg, "region", "")
            })
    return arcs


def optimize_center_trajectories_dual_blank_energy(
        trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        iterations: int = 140,
        attract_weight: float = 0.35,  # 空白区引导引力
        self_repel_weight: float = 0.30,  # 线上节点互斥力
        arc_repel_weight: float = 3.50,  # 归一化微阶梯外推权重
        elastic_weight: float = 0.10,  # 顺向平滑刚度
        transverse_weight: float = 0.60,  # 跨线 Δ 变动均摊刚度
        target_spacing: float = 30.0,
        sample_step: float | None = None,
        influence_radius: float | None = None,
        recompute_blank_every: int = 6,
        resample_every: int = 3,
        anneal: float = 0.96,

        # =========================
        # 新增：基于喷涂均匀性的早停
        # =========================
        uniform_stop: bool = True,
        uniform_metric: str = "cv",  # "cv" / "std" / "var"
        uniform_radius: float | None = None,  # 默认使用 target_spacing d
        uniform_every: int = 3,  # 每几轮计算一次均匀性
        uniform_min_iterations: int = 24,  # 至少迭代 24 轮后才允许早停
        uniform_patience: int = 4,  # 连续 4 次评估变化很小就停止
        uniform_rel_tol: float = 6.18e-4,  # 宽松：允许约 0.5% 的相对变化
        uniform_abs_tol: float = 1e-6,  # 宽松：绝对变化阈值
        uniform_window: int = 10,  # 最近 6 次评估用于检测往返波动
        uniform_oscillation_tol: float = 1.5e-3,  # 宽松：窗口波动范围允许约 3%
        uniform_active_eps: float = 1e-12,  # 认为有效贡献的下限
        uniform_use_area_weights: bool = True,  # 用顶点面积加权，避免顶点密度影响
        uniform_include_fixed: bool = True,  # 是否把 LOWER/UPPER 固定轨迹也算入喷涂贡献
        return_best_uniform: bool = True,  # 早停后返回历史最优轨迹，而不是最后一轮

        on_iteration=None,
) -> List[TrajectorySegment]:
    from scipy.spatial import cKDTree

    d = float(target_spacing)
    step0 = d / 4.0 if sample_step is None else float(sample_step)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("target_spacing must be finite and greater than zero.")
    if not np.isfinite(step0) or step0 <= 0.0:
        raise ValueError("sample_step must be finite and greater than zero.")
    if uniform_metric not in {"cv", "std", "var"}:
        raise ValueError("uniform_metric must be 'cv', 'std', or 'var'.")
    win = int(np.ceil(1.5 * d / step0))

    uniform_every = max(1, int(uniform_every))
    uniform_window = max(4, int(uniform_window))
    uniform_patience = max(1, int(uniform_patience))

    # 1. 提取圆弧的 2D 平面投影几何 (M, u, v, radius)
    arcs = _extract_arc_geometries(trajectory)

    lower_idx = [i for i, s in enumerate(trajectory)
                 if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "LOWER"]
    upper_idx = [i for i, s in enumerate(trajectory)
                 if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "UPPER"]
    center_idx = [i for i, s in enumerate(trajectory)
                  if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "CENTER"]

    if not center_idx:
        return trajectory

    fixed_idx = lower_idx + upper_idx

    fixed_pts = (np.vstack([np.asarray(trajectory[i].xyz, float) for i in fixed_idx])
                 if fixed_idx else np.empty((0, 3)))
    tri_centers = np.asarray(mesh.triangles_center, dtype=float)

    lines = [_resample_polyline(np.asarray(trajectory[i].xyz, float), step0)
             for i in center_idx]
    endpoints = [(L[0].copy(), L[-1].copy()) for L in lines]
    num_lines = len(lines)

    # =========================================================
    # 新增：用于均匀性判断的固定喷涂点、顶点树、顶点面积权重
    # =========================================================
    center_spray_flags = [
        bool(getattr(trajectory[si], "spray_on", True))
        for si in center_idx
    ]

    fixed_spray_pts = np.empty((0, 3), dtype=float)
    if uniform_stop and fixed_idx:
        fblocks = []
        for i in fixed_idx:
            seg = trajectory[i]
            if not bool(getattr(seg, "spray_on", True)):
                continue

            xyz = np.asarray(seg.xyz, dtype=float)
            if len(xyz) >= 2:
                fblocks.append(_resample_polyline(xyz, step0))
            elif len(xyz) == 1:
                fblocks.append(xyz)

        if fblocks:
            fixed_spray_pts = np.vstack(fblocks)

    u_radius = float(d if uniform_radius is None else uniform_radius)
    if u_radius <= 0.0:
        u_radius = d

    n_vertices = len(mesh.vertices)
    vert_tree = None
    vertex_weights = None

    if uniform_stop and n_vertices > 0:
        vert_tree = cKDTree(mesh.vertices)

        if uniform_use_area_weights and len(getattr(mesh, "faces", [])) > 0:
            vertex_weights = np.zeros(n_vertices, dtype=float)
            face_areas = np.asarray(mesh.area_faces, dtype=float)

            np.add.at(vertex_weights, mesh.faces[:, 0], face_areas / 3.0)
            np.add.at(vertex_weights, mesh.faces[:, 1], face_areas / 3.0)
            np.add.at(vertex_weights, mesh.faces[:, 2], face_areas / 3.0)

            if vertex_weights.sum() <= 0.0:
                vertex_weights = np.ones(n_vertices, dtype=float)
        else:
            vertex_weights = np.ones(n_vertices, dtype=float)

    def _spray_field(spray_pts: np.ndarray) -> np.ndarray:
        """
        统计所有喷涂点对 mesh.vertices 的贡献场。

        衰减函数使用 smoothstep:
            w(t) = 1 - 3t^2 + 2t^3

        满足:
            t = 0    -> 1
            t = 0.5  -> 0.5
            t = 1    -> 0
        """
        field = np.zeros(n_vertices, dtype=float)

        if not uniform_stop:
            return field

        if vert_tree is None:
            return field

        if len(spray_pts) == 0:
            return field

        spray_tree = cKDTree(spray_pts)

        # rows: spray points
        # cols: mesh vertices
        coo = spray_tree.sparse_distance_matrix(
            vert_tree,
            u_radius,
            output_type="coo_matrix"
        )

        if coo.nnz == 0:
            return field

        t = np.asarray(coo.data, dtype=float) / u_radius
        t = np.clip(t, 0.0, 1.0)

        # smoothstep 衰减
        w = 1.0 - 3.0 * t * t + 2.0 * t * t * t

        # 数值上接近 0 的可以直接丢掉
        keep = w > 1e-12
        if not np.any(keep):
            return field

        cols = coo.col[keep]
        ww = w[keep]

        field = np.bincount(
            cols,
            weights=ww,
            minlength=n_vertices
        ).astype(float)

        return field

    def _uniform_value(field: np.ndarray) -> float:
        """
        根据顶点贡献场计算均匀性指标。

        默认使用面积加权 CV：
            CV = weighted_std / weighted_mean
        """
        if vertex_weights is None:
            return float("inf")

        # 所有模型顶点都必须参加统计，field == 0 表示该处没有被覆盖。
        # 不能排除零贡献点，否则局部喷得均匀、大片区域漏喷时也可能得到很小的 CV。
        valid = (
                np.isfinite(field)
                & np.isfinite(vertex_weights)
                & (vertex_weights > 0.0)
        )
        if not np.any(valid):
            return float("inf")

        vals = field[valid]
        w = vertex_weights[valid]

        wsum = float(np.sum(w))
        if wsum <= 0.0:
            return float("inf")

        mean = float(np.sum(vals * w) / wsum)
        if mean <= uniform_active_eps:
            return float("inf")

        var = float(np.sum(w * (vals - mean) ** 2) / wsum)
        var = max(0.0, var)
        std = float(np.sqrt(var))

        if uniform_metric == "std":
            return std

        if uniform_metric == "var":
            return var

        return std / mean

    # =========================================================
    # 新增：收敛判断状态
    # =========================================================
    metric_history = []
    smoothed_metric = None
    small_count = 0
    best_metric = float("inf")
    best_lines = None
    evaluations_since_best = 0
    last_uniform_delta = float("nan")
    last_uniform_relative_delta = float("nan")
    last_uniform_threshold = float("nan")
    stop_reason = None

    blanks, void_depth = [], 0.0

    for it in range(iterations):
        P = np.vstack(lines)
        line_id = np.concatenate([np.full(len(L), li, np.int64) for li, L in enumerate(lines)])
        local_id = np.concatenate([np.arange(len(L)) for L in lines])
        offsets = np.r_[0, np.cumsum([len(L) for L in lines])]

        if it % recompute_blank_every == 0:
            blanks, void_depth = _void_centers(tri_centers, fixed_pts, P, d)

        large_void = void_depth > 1.2 * d
        R = float(influence_radius) if influence_radius is not None \
            else (4.5 * d if large_void else 3.0 * d)

        F = np.zeros_like(P)

        # --- A. 空白区引导引力 ---
        for c in blanks:
            dvec = c - P
            dist = np.maximum(np.linalg.norm(dvec, axis=1), 1e-6)
            gate = np.clip(1.0 - (dist / R) ** 2, 0.0, 1.0) ** 2
            F += (attract_weight * gate)[:, None] * (dvec / dist[:, None])

        tc = cKDTree(P)

        # ? --- B. 圆弧影响区 (r ~ r+d) 内的动态阶梯外推 ---
        for arc in arcs:
            m = arc["midpoint"]
            u = arc["u"]
            v = arc["v"]
            r = arc["radius"]

            inside_lines_info = []
            for li in range(num_lines):
                a, b = int(offsets[li]), int(offsets[li + 1])
                curr_L = P[a:b]

                rel = curr_L - m
                x = rel @ u
                y = rel @ v

                r_max = r + 1.0 * d
                mask = (y >= -0.5 * d) & (x ** 2 + y ** 2 <= r_max ** 2)
                if np.any(mask):
                    inside_lines_info.append((li, a, b, curr_L, x, y))

            k = len(inside_lines_info)
            if k == 0:
                continue

            inside_lines_info.sort(
                key=lambda item: float(np.mean(np.linalg.norm(item[3] - m, axis=1)))
            )

            for idx, (li, a, b, curr_L, x, y) in enumerate(inside_lines_info):
                frac = (idx + 1.0) / (k + 1.0)
                r_target = r + frac * 1.0 * d

                valid_x = np.abs(x) <= r_target
                y_target = np.zeros_like(x)
                y_target[valid_x] = np.sqrt(np.maximum(0.0, r_target ** 2 - x[valid_x] ** 2))

                inside_mask = (y >= -0.5 * d) & (x ** 2 + y ** 2 <= r_target ** 2) & (y < y_target)

                F_sub = F[a:b]
                if np.any(inside_mask):
                    depth = np.maximum(0.0, y_target[inside_mask] - y[inside_mask])
                    mag = arc_repel_weight * (3.0 + 10.0 * (depth / d) ** 2)
                    F_sub[inside_mask] += v[None, :] * mag[:, None]

        # 基础 3D 距离微调排斥
        if len(fixed_pts):
            coo = cKDTree(fixed_pts).sparse_distance_matrix(tc, 0.8 * d, output_type="coo_matrix")
            r_, c_, dv = coo.row, coo.col, coo.data
            ok = dv > 1e-6
            r_, c_, dv = r_[ok], c_[ok], dv[ok]
            mag = 0.8 * arc_repel_weight * np.maximum(0.0, (0.8 * d - dv) / d)
            np.add.at(F, c_, (P[c_] - fixed_pts[r_]) / dv[:, None] * mag[:, None])

        # --- C. 同线与局部节点自排斥力 ---
        coo = tc.sparse_distance_matrix(tc, 1.8 * d, output_type="coo_matrix")
        r_, c_, dv = coo.row, coo.col, coo.data
        ok = (dv > 1e-6) & (r_ != c_)
        same = line_id[r_] == line_id[c_]
        ok &= ~(same & (np.abs(local_id[r_] - local_id[c_]) <= win))
        r_, c_, dv = r_[ok], c_[ok], dv[ok]
        mag = self_repel_weight * (1.0 - dv / (1.8 * d))
        np.add.at(F, c_, (P[c_] - P[r_]) / dv[:, None] * mag[:, None])

        # --- D. 沿线纵向顺向平滑 ---
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            if b - a > 2:
                F[a + 1:b - 1] += elastic_weight * (P[a:b - 2] - 2 * P[a + 1:b - 1] + P[a + 2:b])

        # --- E. 跨线网格调和中点场 ---
        if num_lines >= 2:
            line_kdtrees = [cKDTree(L) for L in lines]

            for li in range(1, num_lines - 1):
                a, b = int(offsets[li]), int(offsets[li + 1])
                curr_L = P[a:b]

                _, idx_prev = line_kdtrees[li - 1].query(curr_L)
                _, idx_next = line_kdtrees[li + 1].query(curr_L)

                pt_prev = lines[li - 1][idx_prev]
                pt_next = lines[li + 1][idx_next]

                mid_target = 0.5 * (pt_prev + pt_next)
                F[a:b] += transverse_weight * (mid_target - curr_L)

            # 首尾轨迹线的软约束
            a0, b0 = int(offsets[0]), int(offsets[1])
            _, idx1 = line_kdtrees[1].query(P[a0:b0])
            pt1 = lines[1][idx1]
            d_vec0 = P[a0:b0] - pt1
            dist0 = np.maximum(np.linalg.norm(d_vec0, axis=1), 1e-6)
            F[a0:b0] -= 0.08 * transverse_weight * ((dist0 - d) / d)[:, None] * (d_vec0 / dist0[:, None])

            if num_lines >= 3:
                an, bn = int(offsets[num_lines - 1]), int(offsets[num_lines])
                _, idx_prev_n = line_kdtrees[num_lines - 2].query(P[an:bn])
                pt_prev_n = lines[num_lines - 2][idx_prev_n]
                d_vecn = P[an:bn] - pt_prev_n
                distn = np.maximum(np.linalg.norm(d_vecn, axis=1), 1e-6)
                F[an:bn] -= 0.08 * transverse_weight * ((distn - d) / d)[:, None] * (d_vecn / distn[:, None])

        # --- F. 位移上限控制与退火 ---
        max_step = d * (0.05 + 0.35 * (anneal ** it))
        norm = np.maximum(np.linalg.norm(F, axis=1), 1e-12)
        P += F * np.clip(max_step / norm, 0.0, 1.0)[:, None]

        # 投影回 3D 曲面 + 钉死端点
        P, _, _ = trimesh.proximity.closest_point(mesh, P)
        new_lines = []
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            L = P[a:b]
            L[0], L[-1] = endpoints[li]
            new_lines.append(L)
        lines = new_lines

        if (it + 1) % resample_every == 0:
            lines = [_resample_polyline(L, step0) for L in lines]

        # =========================================================
        # 新增：计算喷涂贡献均匀性，并判断是否停止
        # =========================================================
        uniform_metric_current = None
        should_stop = False

        if (
                uniform_stop
                and vertex_weights is not None
                and ((it + 1) % uniform_every == 0)
        ):
            spray_blocks = []

            if uniform_include_fixed and len(fixed_spray_pts) > 0:
                spray_blocks.append(fixed_spray_pts)

            for li, L in enumerate(lines):
                if center_spray_flags[li]:
                    spray_blocks.append(L)

            if spray_blocks:
                spray_pts = np.vstack(spray_blocks)
            else:
                spray_pts = np.empty((0, 3), dtype=float)

            field = _spray_field(spray_pts)
            metric = _uniform_value(field)
            uniform_metric_current = metric

            if np.isfinite(metric):
                metric_history.append(float(metric))

                # EMA 平滑
                if smoothed_metric is None:
                    smoothed_metric = float(metric)
                    last_uniform_delta = float("nan")
                    last_uniform_relative_delta = float("nan")
                    last_uniform_threshold = float("nan")
                else:
                    prev_smoothed = float(smoothed_metric)
                    smoothed_metric = 0.7 * prev_smoothed + 0.3 * float(metric)

                    delta = abs(smoothed_metric - prev_smoothed)
                    convergence_threshold = (
                            uniform_abs_tol
                            + uniform_rel_tol * abs(prev_smoothed)
                    )
                    last_uniform_delta = float(delta)
                    last_uniform_relative_delta = float(
                        delta / max(abs(prev_smoothed), 1.0e-12)
                    )
                    last_uniform_threshold = float(convergence_threshold)

                    if delta <= convergence_threshold:
                        small_count += 1
                    else:
                        small_count = 0

                # 用原始 CV 保存历史最优轨迹；EMA 只用于判断是否稳定。
                improvement_tol = (
                    uniform_abs_tol
                    + uniform_rel_tol * max(abs(best_metric), 1.0e-12)
                    if np.isfinite(best_metric)
                    else 0.0
                )
                if metric < best_metric - improvement_tol:
                    best_metric = float(metric)
                    best_lines = [L.copy() for L in lines]
                    evaluations_since_best = 0
                else:
                    evaluations_since_best += 1

                # 真正的往返波动：总体漂移很小、窗口振幅不大、且至少反向两次。
                recent = np.asarray(metric_history[-uniform_window:], dtype=float)
                oscillating = False
                relative_drift = float("inf")
                relative_range = float("inf")
                direction_changes = 0
                if len(recent) >= uniform_window:
                    scale = max(abs(float(np.mean(recent))), 1.0e-12)
                    relative_drift = abs(float(recent[-1] - recent[0])) / scale
                    relative_range = float(np.ptp(recent)) / scale
                    differences = np.diff(recent)
                    significant = np.abs(differences) > (
                            uniform_abs_tol + uniform_rel_tol * scale
                    )
                    directions = np.sign(differences[significant])
                    if len(directions) >= 2:
                        direction_changes = int(
                            np.sum(directions[1:] * directions[:-1] < 0.0)
                        )
                    oscillating = (
                            relative_drift <= uniform_oscillation_tol
                            and relative_range <= uniform_oscillation_tol
                            and direction_changes >= 2
                            and evaluations_since_best >= uniform_patience
                    )

                # 前面仍记录指标和历史最佳，但达到最小迭代次数后才允许停止。
                can_stop = (it + 1) >= uniform_min_iterations

                # 停止条件 1：平滑 CV 连续多次几乎不变。
                if can_stop and small_count >= uniform_patience:
                    stop_reason = (
                        f"Uniformity converged at iteration {it + 1}: "
                        f"metric={metric:.6g}, smoothed={smoothed_metric:.6g}, "
                        f"best={best_metric:.6g}"
                    )
                    should_stop = True

                # 停止条件 2：CV 在小范围内反复升降，且已多次没有刷新最优值。
                elif can_stop and oscillating:
                    stop_reason = (
                        f"Uniformity oscillation stabilized at iteration {it + 1}: "
                        f"metric={metric:.6g}, smoothed={smoothed_metric:.6g}, "
                        f"relative_drift={relative_drift:.6g}, "
                        f"relative_range={relative_range:.6g}, "
                        f"direction_changes={direction_changes}, "
                        f"best={best_metric:.6g}"
                    )
                    should_stop = True

        # 原有的回调
        if on_iteration is not None and (
                (it + 1) % 3 == 0 or it == iterations - 1 or should_stop
        ):
            total_energy, energy_components = _trajectory_objective(
                tri_centers=tri_centers,
                fixed_pts=fixed_pts,
                lines=lines,
                target_spacing=d,
                coverage_weight=attract_weight,
                spacing_weight=self_repel_weight + transverse_weight,
                boundary_weight=arc_repel_weight,
                smooth_weight=elastic_weight,
            )

            # 如果 energy_components 是 dict，就把均匀性指标塞进去，方便可视化
            if uniform_stop and metric_history and isinstance(energy_components, dict):
                energy_components = dict(energy_components)
                energy_components["uniform_metric"] = metric_history[-1]
                energy_components["uniform_metric_smoothed"] = smoothed_metric
                energy_components["uniform_metric_best"] = best_metric
                energy_components["uniform_delta"] = last_uniform_delta
                energy_components["uniform_relative_delta"] = last_uniform_relative_delta
                energy_components["uniform_relative_tol"] = uniform_rel_tol
                energy_components["uniform_threshold"] = last_uniform_threshold
                energy_components["uniform_stable_count"] = small_count
                energy_components["uniform_patience"] = uniform_patience
                if uniform_metric_current is not None:
                    energy_components["uniform_metric_current"] = uniform_metric_current
                if stop_reason is not None:
                    energy_components["uniform_stop_reason"] = stop_reason

            preview = list(trajectory)

            for k, si in enumerate(center_idx):
                orig = trajectory[si]
                n_k = len(lines[k])
                # 根据重新采样后的点数赋予前 50% 为 0，后 50% 为 1
                k_phases = np.where(np.arange(n_k) < n_k // 2, 0, 1)
                preview[si] = TrajectorySegment(
                    kind=orig.kind,
                    uv=np.full((n_k, 2), np.nan),
                    xyz=lines[k].copy(),
                    spray_on=orig.spray_on,
                    note=orig.note + " [Optimizing]",
                    region=orig.region,
                    phases=k_phases,  # 【补充】
                )

            on_iteration(preview, it + 1, total_energy, energy_components)

        if should_stop:
            print(f"\n[Uniformity early stop] {stop_reason}")
            break

    # 如果开启了 return_best_uniform，则返回历史最优版本
    if uniform_stop and return_best_uniform and best_lines is not None:
        lines = best_lines

    out = list(trajectory)

    note_suffix = " [Global Arc-Distance Energy Optimized]"
    if stop_reason is not None:
        note_suffix = " [Global Arc-Distance Energy Optimized, Uniformity Early Stop]"

    for k, si in enumerate(center_idx):
        orig = trajectory[si]
        n_k = len(lines[k])
        k_phases = np.where(np.arange(n_k) < n_k // 2, 0, 1)
        out[si] = TrajectorySegment(
            kind=orig.kind,
            uv=np.full((n_k, 2), np.nan),
            xyz=lines[k],
            spray_on=orig.spray_on,
            note=orig.note + note_suffix,
            region=orig.region,
            phases=k_phases,  # 【补充】f
        )

    return out


def generate_seam_aware_trajectory(
        data: ObjUVData,
        seam: PrecutSeam,
        target_spacing: float,
        sample_step: float | None = None,
        overtravel_factor: float = 2.0,
) -> tuple[list[TrajectorySegment], PlanningFrame, dict[str, object]]:
    nominal_spacing = float(target_spacing)
    if nominal_spacing <= 0.0:
        raise ValueError("Trajectory spacing d must be greater than zero.")

    seam_segments = _resolve_main_seam_uv_segments(data, seam)
    frame = _planning_frame(seam_segments)
    local_uv = _to_local(data.uv, frame)
    uv_scale = max(float(np.ptp(local_uv, axis=0).max()), 1.0)
    tolerance = 1.0e-9 * uv_scale
    mapper = _UVToXYZMapper(data)

    lower_vertices = local_uv[local_uv[:, 1] <= tolerance]
    upper_vertices = local_uv[local_uv[:, 1] >= -tolerance]
    if not len(lower_vertices) or not len(upper_vertices):
        raise ValueError("Reference line 1 does not split the UV domain into upper and lower regions.")

    extents = {
        "lower_left": max(0.0, -float(np.min(lower_vertices[:, 0]))),
        "lower_right": max(0.0, float(np.max(lower_vertices[:, 0])) - frame.waist_width),
        "upper_left": max(0.0, -float(np.min(upper_vertices[:, 0]))),
        "upper_right": max(0.0, float(np.max(upper_vertices[:, 0])) - frame.waist_width),
    }

    # =========================================================================
    # 1. 核心改进：选择上、下半区的较窄（较短）一侧作为全局镜像基准侧（0.1mm容差）
    # =========================================================================
    # --- 下半区基准侧选择 ---
    if abs(extents["lower_left"] - extents["lower_right"]) <= 0.1:
        lower_ref_width = extents["lower_left"]
    else:
        lower_ref_width = min(extents["lower_left"], extents["lower_right"])

    # --- 上半区基准侧选择 ---
    if abs(extents["upper_left"] - extents["upper_right"]) <= 0.1:
        upper_ref_width = extents["upper_left"]
    else:
        upper_ref_width = min(extents["upper_left"], extents["upper_right"])

    # =========================================================================
    # 2. 调用 3D 间距优化器（基于腰部、上部基准宽、下部基准宽进行协同优化）
    # =========================================================================
    dimensions = np.asarray([
        frame.waist_width,  # index 0: 中部腰部
        upper_ref_width,  # index 1: 上部基准宽度
        lower_ref_width,  # index 2: 下部基准宽度
    ], dtype=float)

    optimum = optimize_spacing(nominal_spacing, dimensions)
    target_spacing = optimum.spacing

    print(
        "Symmetric Spacing optimization: "
        f"dimensions(center, ref_upper, ref_lower)={dimensions.tolist()}, "
        f"nominal={nominal_spacing:.6g}, optimized={target_spacing:.6g}, "
        f"counts={optimum.counts.tolist()}, "
        f"actual={optimum.actual_spacings.tolist()}."
    )

    sample_step = target_spacing / 4.0 if sample_step is None else float(sample_step)
    if sample_step <= 0.0:
        raise ValueError("Trajectory sampling step must be greater than zero.")
    boundary_snap_tolerance = max(100.0 * (1.0e-9 * uv_scale), 0.5 * sample_step)

    # =========================================================================
    # 3. 强制令左右两侧数量和间距等距镜像对齐
    # =========================================================================
    center_count = int(optimum.counts[0])
    upper_ref_count = int(optimum.counts[1])
    lower_ref_count = int(optimum.counts[2])

    # 下半区左右两侧强制采用等同的参考宽度和参考数量，实现完全对称
    lower_left_count = lower_ref_count
    lower_right_count = lower_ref_count
    lower_left_x, lower_spacing = _balanced_outer_positions(
        lower_ref_width, lower_left_count, 0.0, -1.0
    )
    lower_right_x, _ = _balanced_outer_positions(
        lower_ref_width, lower_right_count, frame.waist_width, 1.0
    )

    # 上半区左右两侧强制采用等同的参考宽度和参考数量，实现完全对称
    upper_left_count = upper_ref_count
    upper_right_count = upper_ref_count
    upper_left_x, upper_spacing = _balanced_outer_positions(
        upper_ref_width, upper_left_count, 0.0, -1.0
    )
    upper_right_x, _ = _balanced_outer_positions(
        upper_ref_width, upper_right_count, frame.waist_width, 1.0
    )

    # 中部腰部区域
    center_x, center_spacing = _balanced_center_positions(
        frame.waist_width, center_count
    )

    # 提取过滤后的有效轨迹（在对称X坐标下，从内向外排序）
    lower_left = _available_tracks(lower_left_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    lower_right = _available_tracks(lower_right_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    upper_left = _available_tracks(upper_left_x, "upper", local_uv, data.uv_faces, tolerance)
    upper_right = _available_tracks(upper_right_x, "upper", local_uv, data.uv_faces, tolerance)

    center_inner_x = center_x

    (
        center_inner,
        rejected_center_boundary_tracks,
        rejected_center_discontinuous_tracks,
    ) = _available_center_tracks(
        center_inner_x, local_uv, data.uv_faces, tolerance
    )
    if not len(center_inner):
        raise ValueError("The waist is too narrow to form an interior central scan.")

    segments: list[TrajectorySegment] = []
    last_surface: TrajectorySegment | None = None
    lower_accepted_x: list[float] = []
    upper_accepted_x: list[float] = []
    rejected_lower_spacing_tracks = 0
    rejected_upper_spacing_tracks = 0

    # 空跑函数
    def append_surface(
            segment: TrajectorySegment,
            connect_from_previous: bool,
            region_transition: bool = False,
    ) -> None:
        nonlocal last_surface
        if connect_from_previous and last_surface is not None:
            if region_transition:
                endpoint_gap = float(np.linalg.norm(segment.xyz[0] - last_surface.xyz[-1]))
                maximum_gap = 12.0 * target_spacing
                if endpoint_gap > maximum_gap:
                    print(
                        f"Warning: large region-transition overtravel gap {endpoint_gap:.6g} "
                        f"(limit {maximum_gap:.6g}) from {last_surface.note!r} -> {segment.note!r}."
                    )
            segments.append(_overtravel_segment(
                last_surface, segment, overtravel_factor * target_spacing
            ))
        segments.append(segment)
        last_surface = segment

    # 切缝连接
    def append_jump(source_uv: np.ndarray, from_a: bool) -> np.ndarray:
        destination_uv = _seam_counterpart(source_uv, seam_segments, from_a=from_a)
        jump_uv = np.vstack([source_uv, destination_uv])
        merged_segment = TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=merged_uv,
            xyz=merged_xyz,
            spray_on=True,
            note=f"{labels[0]} -> {labels[1]}",
            region="LOWER"  # 【新增】
        )
        return destination_uv

    # =========================================================================
    # 4. 下半区排序与共轭连线（基于完美 1-to-1 共轭拓扑）
    # =========================================================================
    paired_lower = min(len(lower_left), len(lower_right))
    for index in range(paired_lower):
        left_track_data = lower_left[index]
        right_track_data = lower_right[index]

        # 探路判定首条轨迹的启动侧，就近连接上一个表面终点
        left_probe = _make_surface_scan(*left_track_data, True, sample_step, frame, mapper)
        right_probe = _make_surface_scan(*right_track_data, True, sample_step, frame, mapper)

        if last_surface is not None:
            left_gap = float(np.linalg.norm(last_surface.xyz[-1] - left_probe.xyz[0]))
            right_gap = float(np.linalg.norm(last_surface.xyz[-1] - right_probe.xyz[0]))
            first_is_left = (left_gap <= right_gap)
        else:
            first_is_left = True  # 首条默认从左侧开始

        if first_is_left:
            master_data, slave_data = left_track_data, right_track_data
            labels = ("lower-left upward (Master)", "lower-right downward (Slave)")
            from_a = True
        else:
            master_data, slave_data = right_track_data, left_track_data
            labels = ("lower-right upward (Master)", "lower-left downward (Slave)")
            from_a = False

        # 生成 Master（自下向上扫至缝隙边）
        first = _make_surface_scan(*master_data, True, sample_step, frame, mapper, note=labels[0])
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=from_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(first.uv, bridge_outside=True)

        # 【核心修正】：直接使用共轭对称线生成向下扫的 Slave 轨，不需要重新搜索和强行扭曲
        second = _make_surface_scan(*slave_data, False, sample_step, frame, mapper, note=labels[1])

        # 强制将 Slave 的起点与跨缝对齐点 destination 重合，确保 3D 过渡无缝合线
        second.uv[0] = destination
        second.xyz = mapper.map(second.uv, bridge_outside=True)

        # 直接在 3D 切缝点将 first 和 second 合并为一条连续的表面喷涂轨迹
        # 1. 合并 Master (first) 和 Slave (second) 的 2D 与 3D 坐标
        # =========================================================================
        # LOWER / UPPER 拼接部分修改示例
        # =========================================================================

        # 1. 生成 Master (first) 和 Slave (second) 各自的 phase
        phase_first = np.zeros(len(first.xyz), dtype=int)  # 0: 前半部分 / Master
        phase_second = np.ones(len(second.xyz), dtype=int)  # 1: 后半部分 / Slave

        # 2. 合并坐标与标识
        merged_uv = np.vstack([first.uv, second.uv])
        merged_xyz = np.vstack([first.xyz, second.xyz])
        merged_phases = np.concatenate([phase_first, phase_second])  # 拼接 phases

        # 3. 进行 3D 去重与平滑（smooth_v_trajectory 会改变点数，需同步处理 phases）
        clean_xyz = smooth_v_trajectory(merged_xyz, min_distance=0.2, smooth_iterations=2)

        # 如果去重改变了点数，根据相对比例重构 phases
        if len(clean_xyz) != len(merged_phases):
            # 前 50% 弧长的点标记为 0，后 50% 标记为 1（或按原始比例计算）
            split_ratio = len(first.xyz) / len(merged_xyz)
            clean_phases = np.where(np.linspace(0.0, 1.0, len(clean_xyz)) < split_ratio, 0, 1)
        else:
            clean_phases = merged_phases

        # 4. 构造带有 phases 标志的 TrajectorySegment
        merged_segment = TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=merged_uv,
            xyz=clean_xyz,
            spray_on=True,
            note=f"{labels[0]} -> {labels[1]}",
            region="LOWER",  # 或 "UPPER"
            phases=clean_phases  # 【传入标识】
        )

        # 4. 追加到主轨迹列表
        append_surface(
            merged_segment, connect_from_previous=(last_surface is not None)
        )

        lower_accepted_x.extend([float(master_data[0]), float(slave_data[0])])

    # =========================================================================
    # 5. 中部腰部排序（保持原有成熟的往返双向择近连接逻辑）
    # =========================================================================
    ascending_tracks = center_inner
    descending_tracks = list(reversed(center_inner))
    inner_tracks = ascending_tracks
    if last_surface is not None and len(center_inner) > 1:
        ascending_first = _make_surface_scan(
            *ascending_tracks[0], upward=True, sample_step=sample_step,
            frame=frame, mapper=mapper,
        )
        descending_first = _make_surface_scan(
            *descending_tracks[0], upward=True, sample_step=sample_step,
            frame=frame, mapper=mapper,
        )
        ascending_gap = float(np.linalg.norm(last_surface.xyz[-1] - ascending_first.xyz[0]))
        descending_gap = float(np.linalg.norm(last_surface.xyz[-1] - descending_first.xyz[0]))
        if descending_gap < ascending_gap:
            inner_tracks = descending_tracks

    for index, track in enumerate(inner_tracks):
        scan = _make_surface_scan(
            *track, upward=(index % 2 == 0), sample_step=sample_step,
            frame=frame, mapper=mapper, note="central bow scan",
            region="CENTER"  # 【新增】
        )
        append_surface(
            scan,
            connect_from_previous=(last_surface is not None),
            region_transition=(index == 0 and last_surface is not None),
        )

    # =========================================================================
    # 6. 上半区排序与共轭连线（与下半区对称，就近起始，1-to-1 共轭接续）
    # =========================================================================
    # =========================================================================
    # 6. 上半区排序与共轭连线（与下半区对称，就近起始，1-to-1 共轭接续）
    # =========================================================================
    paired_upper = min(len(upper_left), len(upper_right))
    for index in range(paired_upper):
        left_track_data = upper_left[index]
        right_track_data = upper_right[index]

        # 探路判定首条轨迹启动侧，优先连接中间区最后一个终点
        left_probe = _make_surface_scan(*left_track_data, False, sample_step, frame, mapper)
        right_probe = _make_surface_scan(*right_track_data, False, sample_step, frame, mapper)

        if last_surface is not None:
            left_gap = float(np.linalg.norm(last_surface.xyz[-1] - left_probe.xyz[0]))
            right_gap = float(np.linalg.norm(last_surface.xyz[-1] - right_probe.xyz[0]))
            first_is_left = (left_gap <= right_gap)
        else:
            first_is_left = True

        if first_is_left:
            master_data, slave_data = left_track_data, right_track_data
            labels = ("upper-left downward (Master)", "upper-right upward (Slave)")
            from_a = True
        else:
            master_data, slave_data = right_track_data, left_track_data
            labels = ("upper-right downward (Master)", "upper-left upward (Slave)")
            from_a = False

        # 生成 Master（自上向下扫至缝隙边）
        first = _make_surface_scan(*master_data, False, sample_step, frame, mapper, note=labels[0])
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=from_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(first.uv, bridge_outside=True)

        # 【核心修正】：直接使用共轭对称坐标生成向上扫的 Slave 轨
        second = _make_surface_scan(*slave_data, True, sample_step, frame, mapper, note=labels[1])

        # 强制首尾对齐
        second.uv[0] = destination
        second.xyz = mapper.map(second.uv, bridge_outside=True)

        # ======================= 【修改开始】 =======================
        # 将 Master 和 Slave 合并为一段连续的 SURFACE_SCAN，不再插入 SEAM_JUMP
        # 生成 Master 和 Slave 各自的 phase
        phase_first = np.zeros(len(first.xyz), dtype=int)  # 0: 前半部分 / Master
        phase_second = np.ones(len(second.xyz), dtype=int)  # 1: 后半部分 / Slave

        merged_uv = np.vstack([first.uv, second.uv])
        merged_xyz = np.vstack([first.xyz, second.xyz])
        merged_phases = np.concatenate([phase_first, phase_second])  # 【新增】拼接 phases

        merged_segment = TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=merged_uv,
            xyz=merged_xyz,
            spray_on=True,
            note=f"{labels[0]} -> {labels[1]}",
            region="UPPER",
            phases=merged_phases  # 【补充传入 phases】
        )
        # 一次性将合并后的整段轨迹加入列表
        append_surface(
            merged_segment,
            connect_from_previous=(last_surface is not None),
            region_transition=(not upper_accepted_x and last_surface is not None),
        )
        # ======================= 【修改结束】 =======================

        upper_accepted_x.extend([float(master_data[0]), float(slave_data[0])])

    # =========================================================================
    # 7. 3D 跨缝平滑过渡处理与元数据构建（保持原有机制）
    # =========================================================================
    # segments = _smooth_seam_jumps_xyz(
    #     segments, nominal_radius=0.5 * target_spacing, sample_step=sample_step,
    # )

    metadata: dict[str, object] = {
        "requested_spacing": nominal_spacing,
        "optimized_spacing": target_spacing,
        "optimization_dimensions": dimensions.tolist(),
        "optimization_counts": optimum.counts.tolist(),
        "optimization_actual_spacings": optimum.actual_spacings.tolist(),
        "optimization_objective": optimum.objective,
        "optimization_curve": {
            "spacing": optimum.candidates.tolist(),
            "objective": optimum.objective_values.tolist(),
        },
        "sample_step": sample_step,
        "uv_gap_policy": "bridge_complete_offset_by_xyz_interpolation",
        "seam_xyz_smoothing": "G1_cubic_coordinated_motion",
        "seam_xyz_nominal_radius": 0.5 * target_spacing,
        "degenerate_waist_halves": [],
        "overtravel": overtravel_factor * target_spacing,
        "boundary_snap_tolerance": boundary_snap_tolerance,
        "waist_pair_index": frame.waist_pair_index,
        "waist_width": frame.waist_width,
        "reference_origin_uv": frame.origin.tolist(),
        "reference_x_axis_uv": frame.x_axis.tolist(),
        "reference_y_axis_uv": frame.y_axis.tolist(),
        "extents": extents,
        "actual_spacings": {
            "lower_left": lower_spacing,
            "lower_right": lower_spacing,
            "center": center_spacing,
            "upper_left": upper_spacing,
            "upper_right": upper_spacing,
        },
        "track_counts": {
            "lower_each_side": paired_lower,
            "center_inner": len(center_inner),
            "center_rejected_seam_boundary": rejected_center_boundary_tracks,
            "center_rejected_discontinuous": rejected_center_discontinuous_tracks,
            "center_waist_half_scans": 0,
            "upper_each_side": paired_upper,
            "lower_rejected_spacing": rejected_lower_spacing_tracks,
            "upper_rejected_spacing": rejected_upper_spacing_tracks,
        },
        "segment_count": len(segments),
    }
    return segments, frame, metadata


def _export_trajectory(
        csv_path: Path,
        metadata_path: Path,
        segments: Sequence[TrajectorySegment],
        metadata: dict[str, object],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "segment_id", "point_id", "event", "spray_on", "u", "v",
            "x", "y", "z", "note",
        ])
        for segment_id, segment in enumerate(segments):
            for point_id, xyz in enumerate(segment.xyz):
                uv = segment.uv[point_id] if point_id < len(segment.uv) else np.array([np.nan, np.nan])
                writer.writerow([
                    segment_id, point_id, segment.kind, int(segment.spray_on),
                    "" if not np.isfinite(uv[0]) else f"{uv[0]:.12g}",
                    "" if not np.isfinite(uv[1]) else f"{uv[1]:.12g}",
                    f"{xyz[0]:.12g}", f"{xyz[1]:.12g}", f"{xyz[2]:.12g}",
                    segment.note,
                ])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved trajectory CSV: {csv_path}")
    print(f"Saved trajectory metadata: {metadata_path}")


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale

    # 【已修改】：将原本的 [qx, qy, qz, qw] 调整为 [qw, qx, qy, qz]
    quat = np.asarray([qw, qx, qy, qz], dtype=float)
    return quat / max(float(np.linalg.norm(quat)), EPS)


def _orientation_quaternion(tangent: np.ndarray, inward_normal: np.ndarray) -> np.ndarray:
    # Y-axis: 轨迹前进方向 (Tangent)
    y_axis = np.asarray(tangent, dtype=float)
    y_norm = np.linalg.norm(y_axis)
    if y_norm > EPS:
        y_axis /= y_norm
    else:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    # Z-axis: 指向模型内部的法向 (Inward Normal)
    z_axis = np.asarray(inward_normal, dtype=float)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > EPS:
        z_axis /= z_norm
    else:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    # X-axis: Y x Z (右手系叉乘)
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= EPS:
        # 处理切向与法向共线的奇异情况
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(helper, z_axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = np.cross(helper, z_axis)
        x_norm = np.linalg.norm(x_axis)

    x_axis /= max(x_norm, EPS)

    # 重新正交化 Y-axis 以保证帧的严密正交性: Y = Z x X
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), EPS)

    # 构造旋转矩阵 col(0)=X, col(1)=Y, col(2)=Z
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return _quaternion_from_matrix(rotation_matrix)


def _export_paq_trajectory(
        path: Path,
        mesh: trimesh.Trimesh,
        segments: Sequence[TrajectorySegment],
        speed: float,
        spray_distance: float,
) -> None:
    points = [xyz for segment in segments if len(segment.xyz) for xyz in segment.xyz]
    if not points:
        path.write_text("", encoding="utf-8")
        return

    surface_points = np.asarray(points, dtype=float)
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, surface_points)
    normals = np.asarray(mesh.face_normals, dtype=float)[np.asarray(face_ids, dtype=np.int64)]
    inward_normals = -normals

    # 位置点偏置使用指向外部的正常面法向 normals
    export_points = closest + normals * float(spray_distance)

    # =========================================================================
    # 核心修改：分段计算切向，并根据 Z 轴走向将方向统一朝上（Z 轴正方向）
    # =========================================================================
    # 1. 统计每个有效段的点数
    segment_lengths = [len(seg.xyz) for seg in segments if len(seg.xyz) > 0]

    # 2. 将扁平化的 export_points 重新切分为与原 segment 一一对应的子数组
    split_indices = np.cumsum(segment_lengths)[:-1]
    segment_export_points = np.split(export_points, split_indices)

    segment_tangents = []
    for pts in segment_export_points:
        n_pts = len(pts)
        if n_pts == 0:
            continue

        # 初始化当前段的切向数组
        t = np.empty_like(pts)
        if n_pts == 1:
            t[0] = np.array([1.0, 0.0, 0.0])
        else:
            # 基础切向计算（底层、中层、顶层点逻辑）
            t[0] = pts[1] - pts[0]
            t[-1] = pts[-1] - pts[-2]
            t[1:-1] = pts[2:] - pts[:-2]

            # 统一方向判定：若该段终点比起点矮（dz < 0，即物理向下行进），
            # 则反转其所有点的切向，使其指向不变（即下一个点指向前一个点，依然朝上）
            dz = pts[-1, 2] - pts[0, 2]
            if dz < 0:
                t = -t

        segment_tangents.append(t)

    # 3. 将各段切向重新合并为与 export_points 一一对应的数组
    tangents = np.vstack(segment_tangents)
    # =========================================================================

    cumulative = np.zeros(len(export_points), dtype=float)
    if len(export_points) > 1:
        cumulative[1:] = np.cumsum(np.linalg.norm(export_points[1:] - export_points[:-1], axis=1))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for point, tangent, normal, distance in zip(export_points, tangents, inward_normals, cumulative):
            quat = _orientation_quaternion(tangent, normal)
            stream.write(
                f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f} "
                f"{quat[0]:.3f} {quat[1]:.3f} {quat[2]:.3f} {quat[3]:.3f} "
                f"{float(speed):.3f} 0.000 {distance:.6f} 0 {float(spray_distance):.3f}\n"
            )
    print(f"Saved PAQ trajectory: {path}")


def _read_obj_cut_edges(path: Path, uv_tolerance: float = 1.0e-8) -> np.ndarray:
    vertices = []
    texture_coordinates = []
    faces = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("vt "):
                values = line.split()
                texture_coordinates.append([float(values[1]), float(values[2])])
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    continue
                parsed = []
                for corner in corners:
                    fields = corner.split("/")
                    if len(fields) < 2 or not fields[0] or not fields[1]:
                        continue
                    parsed.append((int(fields[0]) - 1, int(fields[1]) - 1))
                if len(parsed) == 3:
                    faces.append(parsed)

    xyz = np.asarray(vertices, dtype=float)
    uv = np.asarray(texture_coordinates, dtype=float)
    if not len(xyz) or not len(uv) or not faces:
        return np.empty((0, 2, 3), dtype=float)

    edge_faces: dict[tuple[int, int], tuple[int, int]] = {}
    seam_edges: list[np.ndarray] = []
    uv_span = max(float(np.ptp(uv, axis=0).max()), 1.0)
    tolerance = uv_tolerance * uv_span
    for face in faces:
        for local in range(3):
            first = face[local]
            second = face[(local + 1) % 3]
            vertex_a, vertex_b = first[0], second[0]
            key = (min(vertex_a, vertex_b), max(vertex_a, vertex_b))
            uv_a, uv_b = first[1], second[1]
            canonical_uv = (uv_a, uv_b) if vertex_a < vertex_b else (uv_b, uv_a)
            if key not in edge_faces:
                edge_faces[key] = canonical_uv
                continue
            previous_uv_a, previous_uv_b = edge_faces[key]
            uv_a, uv_b = canonical_uv
            if (
                    previous_uv_a >= len(uv)
                    or previous_uv_b >= len(uv)
                    or uv_a >= len(uv)
                    or uv_b >= len(uv)
            ):
                continue
            if (
                    np.linalg.norm(uv[previous_uv_a] - uv[uv_a]) > tolerance
                    or np.linalg.norm(uv[previous_uv_b] - uv[uv_b]) > tolerance
            ):
                if vertex_a < len(xyz) and vertex_b < len(xyz):
                    seam_edges.append(xyz[[vertex_a, vertex_b]])

    return np.asarray(seam_edges, dtype=float) if seam_edges else np.empty((0, 2, 3), dtype=float)


def _read_iteration_obj(path: Path):
    vertices = []
    uv = []
    vertex_faces = []
    uv_faces = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("vt "):
                values = line.split()
                uv.append([float(values[1]), float(values[2])])
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    continue
                vertex_face = []
                uv_face = []
                for corner in corners:
                    indices = corner.split("/")
                    vertex_face.append(int(indices[0]) - 1)
                    uv_face.append(int(indices[1]) - 1)
                vertex_faces.append(vertex_face)
                uv_faces.append(uv_face)
    return (np.asarray(vertices, dtype=float), np.asarray(vertex_faces, dtype=np.int64),
            np.asarray(uv, dtype=float), np.asarray(uv_faces, dtype=np.int64))


# =========================================================================
#                    VTK & Tkinter Interactive Visualizer Module
# =========================================================================

def vtk_mesh(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for point in vertices:
        points.InsertNextPoint(*map(float, point))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for vertex_id in face:
            cells.InsertCellPoint(int(vertex_id))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def vtk_polyline(points_array: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for point in points_array:
        point3 = np.r_[point, np.zeros(max(0, 3 - len(point)))]
        points.InsertNextPoint(*map(float, point3[:3]))
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points_array))
    for index in range(len(points_array)):
        lines.InsertCellPoint(index)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata


def vtk_points(points_array: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    for pt in points_array:
        points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0)
    vertices = vtk.vtkCellArray()
    for i in range(len(points_array)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    return polydata


def vtk_actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float], opacity: float = 0.90,
              width: float = 1.0, wireframe: bool = False) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # 【防 Z-fighting 增强】：强制 VTK 在光栅化时给线/面添加微小的深度偏移
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    # 针对线段的特定偏移参数（将线微微向摄像机方向拉出）
    try:
        mapper.SetResolveCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
    except AttributeError:
        pass  # 兼容旧版 VTK

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)

    # 【视觉增强】：将线段渲染为实体细管，能大幅减少共面闪烁，且视觉效果更高级
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetRenderLinesAsTubes(True)

    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
    return actor


# 1. 修复 VTK 9.5 弃用警告的 add_title 函数
def add_title(renderer: vtk.vtkRenderer, text: str) -> None:
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.SetPosition(18, 18)
    actor.GetTextProperty().SetFontSize(16)
    actor.GetTextProperty().SetColor(0.12, 0.12, 0.12)
    renderer.AddViewProp(actor)  # 改为 AddViewProp，消除 DeprecationWarning


def add_penalty_chart(renderer: vtk.vtkRenderer, candidates: np.ndarray, objectives: np.ndarray,
                      optimized_d: float, optimized_f: float) -> vtk.vtkContextActor:
    table = vtk.vtkTable()
    d_values, f_values = vtk.vtkDoubleArray(), vtk.vtkDoubleArray()
    d_values.SetName("d")
    f_values.SetName("F(d)")
    for d, penalty in zip(candidates, objectives):
        d_values.InsertNextValue(float(d))
        f_values.InsertNextValue(float(penalty))
    table.AddColumn(d_values)
    table.AddColumn(f_values)

    chart = vtk.vtkChartXY()
    chart.SetTitle(f"Penalty: d*={optimized_d:.6g}, F(d*)={optimized_f:.6g}")
    curve = chart.AddPlot(vtk.vtkChart.LINE)
    curve.SetInputData(table, 0, 1)
    curve.GetPen().SetColor(25, 72, 205, 255)
    curve.GetPen().SetWidth(2.5)

    point_table = vtk.vtkTable()
    best_d, best_f = vtk.vtkDoubleArray(), vtk.vtkDoubleArray()
    best_d.SetName("d*")
    best_f.SetName("F(best)")
    best_d.InsertNextValue(optimized_d)
    best_f.InsertNextValue(optimized_f)
    point_table.AddColumn(best_d)
    point_table.AddColumn(best_f)
    marker = chart.AddPlot(vtk.vtkChart.POINTS)
    marker.SetInputData(point_table, 0, 1)
    marker.SetColor(220, 20, 35, 255)
    marker.SetWidth(10.0)
    marker.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)

    x_axis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
    y_axis = chart.GetAxis(vtk.vtkAxis.LEFT)
    x_axis.SetTitle("candidate spacing d")
    y_axis.SetTitle("penalty F(d)")
    x_axis.SetPrecision(5)
    y_axis.SetNotation(vtk.vtkAxis.SCIENTIFIC_NOTATION)
    y_axis.SetPrecision(4)
    fmin, fmax = float(objectives.min()), float(objectives.max())
    padding = max(0.04 * (fmax - fmin), max(abs(fmin), abs(fmax), EPS) * 1e-4)
    x_axis.SetBehavior(vtk.vtkAxis.FIXED)
    x_axis.SetRange(float(candidates.min()), float(candidates.max()))
    y_axis.SetBehavior(vtk.vtkAxis.FIXED)
    y_axis.SetRange(fmin - padding, fmax + padding)
    context = vtk.vtkContextActor()
    context.GetScene().AddItem(chart)
    renderer.AddActor(context)
    return context


def add_energy_history_chart(
        renderer: vtk.vtkRenderer,
        iterations: Sequence[int],
        energies: Sequence[float],
) -> vtk.vtkContextActor:
    """Draw the total-energy history accumulated during live optimization."""
    table = vtk.vtkTable()
    iteration_values = vtk.vtkDoubleArray()
    energy_values = vtk.vtkDoubleArray()
    iteration_values.SetName("iteration")
    energy_values.SetName("total objective E")
    for iteration, energy in zip(iterations, energies):
        iteration_values.InsertNextValue(float(iteration))
        energy_values.InsertNextValue(float(energy))
    table.AddColumn(iteration_values)
    table.AddColumn(energy_values)

    chart = vtk.vtkChartXY()
    latest = float(energies[-1])
    chart.SetTitle(f"Trajectory objective trend: E={latest:.6g}")
    curve = chart.AddPlot(vtk.vtkChart.LINE)
    curve.SetInputData(table, 0, 1)
    curve.GetPen().SetColor(220, 20, 35, 255)
    curve.GetPen().SetWidth(2.5)

    marker = chart.AddPlot(vtk.vtkChart.POINTS)
    marker.SetInputData(table, 0, 1)
    marker.SetColor(25, 72, 205, 255)
    marker.SetWidth(5.0)
    marker.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)

    x_axis = chart.GetAxis(vtk.vtkAxis.BOTTOM)
    y_axis = chart.GetAxis(vtk.vtkAxis.LEFT)
    x_axis.SetTitle("iteration")
    y_axis.SetTitle("total objective E")
    x_axis.SetPrecision(0)
    y_axis.SetNotation(vtk.vtkAxis.SCIENTIFIC_NOTATION)
    y_axis.SetPrecision(4)

    iteration_min = float(min(iterations))
    iteration_max = float(max(iterations))
    if iteration_max <= iteration_min:
        iteration_min -= 1.0
        iteration_max += 1.0
    energy_min = float(min(energies))
    energy_max = float(max(energies))
    energy_padding = max(
        0.06 * (energy_max - energy_min),
        max(abs(energy_min), abs(energy_max), EPS) * 1.0e-3,
    )
    x_axis.SetBehavior(vtk.vtkAxis.FIXED)
    x_axis.SetRange(iteration_min, iteration_max)
    y_axis.SetBehavior(vtk.vtkAxis.FIXED)
    y_axis.SetRange(energy_min - energy_padding, energy_max + energy_padding)

    context = vtk.vtkContextActor()
    context.GetScene().AddItem(chart)
    renderer.AddActor(context)
    return context


class InteractiveVisualizer:
    def __init__(self, mesh: trimesh.Trimesh, data: ObjUVData, precut_seam: PrecutSeam,
                 cut_edges: np.ndarray, sample_step: float, initial_spacing: float,
                 csv_path: Path, metadata_path: Path, paq_path: Path,
                 trajectory_speed: float, spray_distance: float):
        self.mesh = mesh
        self.data = data
        self.precut_seam = precut_seam
        self.cut_edges = cut_edges
        self.sample_step = sample_step
        self.current_spacing = initial_spacing
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.paq_path = paq_path
        self.trajectory_speed = trajectory_speed
        self.spray_distance = spray_distance
        self.current_trajectory: list[TrajectorySegment] = []
        self.current_metadata: dict[str, object] | None = None
        self.a_max = 1000.0  # 默认值
        self.j_max = 5000.0  # 默认值
        self.hide_model = False
        self.hide_traj = False

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("OptCuts trajectory - VTK/OpenGL (Interactive)")
        self.window.SetSize(1600, 900)
        # 【新增】：1. 设置渲染窗口支持 2 个层级 (Layer 0 和 Layer 1)
        self.window.SetNumberOfLayers(2)

        # 底层渲染器 Layer 0：专门绘制 3D 网格模型
        self.left_renderer = vtk.vtkRenderer()
        self.left_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
        self.left_renderer.SetLayer(0)

        # 【新增】：2. 定义顶层渲染器 Layer 1（专门置顶绘制 3D 轨迹）
        self.left_overlay_renderer = vtk.vtkRenderer()
        self.left_overlay_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
        self.left_overlay_renderer.SetLayer(1)
        self.left_overlay_renderer.EraseOff()  # 透明背景
        # 共享 3D 视角相机（旋转/平移/缩放自动完美同步）
        self.left_overlay_renderer.SetActiveCamera(self.left_renderer.GetActiveCamera())

        self.right_renderer = vtk.vtkRenderer()
        self.right_renderer.SetViewport(0.5, 0.34, 1.0, 1.0)
        self.right_renderer.SetLayer(0)

        self.chart_renderer = vtk.vtkRenderer()
        self.chart_renderer.SetViewport(0.5, 0.0, 1.0, 0.34)
        self.chart_renderer.SetLayer(0)

        for ren in (self.left_renderer, self.left_overlay_renderer, self.right_renderer, self.chart_renderer):
            if ren != self.left_overlay_renderer:
                ren.SetBackground(0.96, 0.97, 0.98)
            self.window.AddRenderer(ren)

        self.vertices_3d = np.asarray(self.mesh.vertices, float)
        self.faces_3d = np.asarray(self.mesh.faces, int)

        self.mesh_actor_3d = vtk_actor(vtk_mesh(self.vertices_3d, self.faces_3d), (0.38, 0.62, 0.78), opacity=1.0)
        self.left_renderer.AddActor(self.mesh_actor_3d)

        uv3 = np.column_stack([self.data.uv, np.zeros(len(self.data.uv))])
        self.mesh_actor_2d = vtk_actor(vtk_mesh(uv3, self.data.uv_faces), (0.62, 0.66, 0.70), opacity=0.30,
                                       wireframe=True)
        self.right_renderer.AddActor(self.mesh_actor_2d)

        add_title(self.left_renderer, "3D seam-aware spray trajectory")
        add_title(self.right_renderer, "2D OptCuts UV trajectory")

        self.trajectory_actors_3d = []
        self.trajectory_actors_2d = []
        self.chart_context_actor = None
        self.text_actor_2d_info = None
        self.energy_iterations: list[int] = []
        self.energy_values: list[float] = []

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def apply_visibility(self):
        """Enforce standard visibility rules across mesh and trajectory elements."""
        self.mesh_actor_3d.SetVisibility(not self.hide_model)
        self.mesh_actor_2d.SetVisibility(not self.hide_model)
        for actor in self.trajectory_actors_3d:
            actor.SetVisibility(not self.hide_traj)
        for actor in self.trajectory_actors_2d:
            actor.SetVisibility(not self.hide_traj)
        self.window.Render()

    def reset_energy_history(self) -> None:
        self.energy_iterations.clear()
        self.energy_values.clear()

    def record_energy(self, iteration: int, total_energy: float) -> None:
        self.energy_iterations.append(int(iteration))
        self.energy_values.append(float(total_energy))

    def update_trajectory(self, trajectory: Sequence[TrajectorySegment], planning_frame: PlanningFrame,
                          metadata: dict[str, object],
                          trajectory_2d: Sequence[TrajectorySegment] | None = None):
        self.current_trajectory = list(trajectory)
        self.current_metadata = metadata
        display_2d = trajectory_2d if trajectory_2d is not None else trajectory

        # 1. 清理原有的 3D 和 2D 轨迹 Actor
        for actor in self.trajectory_actors_3d:
            self.left_overlay_renderer.RemoveActor(actor)
        self.trajectory_actors_3d.clear()

        for actor in self.trajectory_actors_2d:
            self.right_renderer.RemoveActor(actor)
        self.trajectory_actors_2d.clear()

        if self.chart_context_actor:
            self.chart_renderer.RemoveActor(self.chart_context_actor)

        if self.text_actor_2d_info:
            self.right_renderer.RemoveActor(self.text_actor_2d_info)

        # === 【3D 轨迹绘制 - Actor 合并极速渲染（添加到 Layer 1 顶层渲染器）】 ===
        scan_append_3d = vtk.vtkAppendPolyData()
        overtravel_append_3d = vtk.vtkAppendPolyData()
        blend_append_3d = vtk.vtkAppendPolyData()

        has_scan = False
        has_overtravel = False
        has_blend = False

        for seg in trajectory:
            if len(seg.xyz) < 2:
                continue
            poly = vtk_polyline(seg.xyz)
            if seg.kind == "SURFACE_SCAN":
                scan_append_3d.AddInputData(poly)
                has_scan = True
            elif seg.kind == "OVERTRAVEL":
                overtravel_append_3d.AddInputData(poly)
                has_overtravel = True
            elif seg.kind == "SEAM_BLEND_3D":
                blend_append_3d.AddInputData(poly)
                has_blend = True

        # 绘制主喷涂轨迹 (红色)
        if has_scan:
            scan_append_3d.Update()
            actor = vtk_actor(scan_append_3d.GetOutput(), (0.86, 0.05, 0.08), width=3.5)
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        # 绘制过渡空跑轨迹 (绿色)
        if has_overtravel:
            overtravel_append_3d.Update()
            actor = vtk_actor(overtravel_append_3d.GetOutput(), (0.0, 0.8, 0.0), width=2.5)
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        # 绘制接缝平滑轨迹 (橙色)
        if has_blend:
            blend_append_3d.Update()
            actor = vtk_actor(blend_append_3d.GetOutput(), (1.0, 0.5, 0.0), width=3.0)
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        # === 【2D 轨迹绘制 - Actor 合并】 ===
        scan_append_2d = vtk.vtkAppendPolyData()
        has_scan_2d = False

        for seg in display_2d:
            if seg.kind != "SURFACE_SCAN":
                continue
            valid_uv = [uv for uv in seg.uv if np.isfinite(uv).all()]
            if len(valid_uv) < 2:
                continue

            uv_arr = np.asarray(valid_uv)
            dists = np.linalg.norm(np.diff(uv_arr, axis=0), axis=1)
            max_jump = 5.0 * self.sample_step
            split_indices = np.where(dists > max_jump)[0] + 1
            sub_lines = np.split(uv_arr, split_indices)

            for line in sub_lines:
                if len(line) >= 2:
                    scan_append_2d.AddInputData(vtk_polyline(line))
                    has_scan_2d = True

        if has_scan_2d:
            scan_append_2d.Update()
            actor_2d = vtk_actor(scan_append_2d.GetOutput(), (0.86, 0.05, 0.08), width=3.0)
            self.right_renderer.AddActor(actor_2d)
            self.trajectory_actors_2d.append(actor_2d)

        # === 3. 绘制辅助参考线与点集 ===
        ref_line_pts = np.vstack([
            planning_frame.origin,
            planning_frame.origin + planning_frame.waist_width * planning_frame.x_axis
        ])
        ref_actor = vtk_actor(vtk_polyline(ref_line_pts), (0.0, 0.8, 0.8), width=3.5)
        self.right_renderer.AddActor(ref_actor)
        self.trajectory_actors_2d.append(ref_actor)

        cut_uv_pts = []
        for pair in self.precut_seam.cut_vertex_pairs:
            for idx in pair:
                if idx < len(self.data.uv):
                    cut_uv_pts.append(self.data.uv[idx])
        if cut_uv_pts:
            pts_actor = vtk_actor(vtk_points(np.asarray(cut_uv_pts)), (0.1, 0.4, 0.8))
            pts_actor.GetProperty().SetPointSize(8.0)
            self.right_renderer.AddActor(pts_actor)
            self.trajectory_actors_2d.append(pts_actor)

        # === 4. 更新散点图和数据文本 ===
        if self.energy_values:
            self.chart_context_actor = add_energy_history_chart(
                self.chart_renderer, self.energy_iterations, self.energy_values
            )
        else:
            opt_curve = metadata["optimization_curve"]
            candidates = np.asarray(opt_curve["spacing"])
            objectives = np.asarray(opt_curve["objective"])
            opt_d = metadata["optimized_spacing"]
            opt_f = metadata["optimization_objective"]
            self.chart_context_actor = add_penalty_chart(
                self.chart_renderer, candidates, objectives, opt_d, opt_f
            )

        self.text_actor_2d_info = vtk.vtkTextActor()
        actual_spacings = metadata["actual_spacings"]
        info_text = f"Actual spacing  center: {actual_spacings['center']:.4f}  upper: {actual_spacings['upper_left']:.4f}  lower: {actual_spacings['lower_left']:.4f}"
        self.text_actor_2d_info.SetInput(info_text)
        self.text_actor_2d_info.SetPosition(400, 18)
        self.text_actor_2d_info.GetTextProperty().SetFontSize(18)
        self.text_actor_2d_info.GetTextProperty().SetColor(0.12, 0.12, 0.12)
        self.right_renderer.AddViewProp(self.text_actor_2d_info)

        self.apply_visibility()

    def reset_cameras(self):
        center = self.vertices_3d.mean(axis=0)
        _, _, basis = np.linalg.svd(self.vertices_3d - center, full_matrices=False)
        normal, view_up = basis[2], basis[1]
        distance = max(float(np.linalg.norm(np.ptp(self.vertices_3d, axis=0))), 1.0)

        camera = self.left_renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + normal * 2.0 * distance))
        camera.SetViewUp(*view_up)
        self.left_renderer.ResetCamera()

        camera2 = self.right_renderer.GetActiveCamera()
        camera2.ParallelProjectionOn()
        camera2.SetPosition(0.5, 0.5, 1)
        camera2.SetFocalPoint(0.5, 0.5, 0)
        camera2.SetViewUp(0, 1, 0)
        self.right_renderer.ResetCamera()


class TkControlPanel:
    def __init__(self, visualizer: InteractiveVisualizer):  # 只需要接收 1 个 visualizer 参数
        self.visualizer = visualizer
        self.root = tk.Tk()
        self.root.title("控制台 - 属性控制")
        self.root.geometry("400x330+50+50")
        self.root.wm_attributes("-topmost", True)

        # Spacing update entry controls
        tk.Label(self.root, text="目标间距 d: ", font=("Arial", 11)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.spacing_var = tk.StringVar(value=f"{self.visualizer.current_spacing:.12g}")
        self.entry = tk.Entry(self.root, textvariable=self.spacing_var, width=15, font=("Arial", 11))
        self.entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        tk.Label(self.root, text="轨迹速度: ", font=("Arial", 11)).grid(row=1, column=0, padx=10, pady=6, sticky="w")
        self.speed_var = tk.StringVar(value=f"{self.visualizer.trajectory_speed:.12g}")
        self.speed_entry = tk.Entry(self.root, textvariable=self.speed_var, width=15, font=("Arial", 11))
        self.speed_entry.grid(row=1, column=1, padx=10, pady=6, sticky="ew")

        tk.Label(self.root, text="喷涂距离: ", font=("Arial", 11)).grid(row=2, column=0, padx=10, pady=6, sticky="w")

        self.spray_distance_var = tk.StringVar(value=f"{self.visualizer.spray_distance:.12g}")
        self.spray_distance_entry = tk.Entry(
            self.root, textvariable=self.spray_distance_var, width=15, font=("Arial", 11)
        )
        self.spray_distance_entry.grid(row=2, column=1, padx=10, pady=6, sticky="ew")

        tk.Label(self.root, text="最大加速度 (mm/s²): ", font=("Arial", 11)).grid(row=3, column=0, padx=10, pady=6,
                                                                                  sticky="w")
        self.a_max_var = tk.StringVar(value="1000.0")  # 默认值，根据实际机器人设定
        self.a_max_entry = tk.Entry(self.root, textvariable=self.a_max_var, width=15, font=("Arial", 11))
        self.a_max_entry.grid(row=3, column=1, padx=10, pady=6, sticky="ew")

        # 【新增】：最大加加速度输入
        tk.Label(self.root, text="最大加加速度 (mm/s³): ", font=("Arial", 11)).grid(row=4, column=0, padx=10, pady=6,
                                                                                    sticky="w")
        self.j_max_var = tk.StringVar(value="5000.0")  # 默认值
        self.j_max_entry = tk.Entry(self.root, textvariable=self.j_max_var, width=15, font=("Arial", 11))
        self.j_max_entry.grid(row=4, column=1, padx=10, pady=6, sticky="ew")

        # 原有的按钮行号下移
        self.btn = tk.Button(self.root, text="更新轨迹", command=self.update_spacing, font=("Arial", 11), bg="#1948CD",
                             fg="white")
        self.btn.grid(row=5, column=0, padx=10, pady=5, sticky="ew")

        self.export_btn = tk.Button(self.root, text="导出轨迹", command=self.export_trajectory, font=("Arial", 11),
                                    bg="#1948CD", fg="white")
        self.export_btn.grid(row=5, column=1, padx=10, pady=5, sticky="ew")

        # Show/Hide checkboxes
        self.hide_model_var = tk.BooleanVar(value=False)
        self.cb_model = tk.Checkbutton(
            self.root, text="隐藏三维网格模型", variable=self.hide_model_var, command=self.toggle_model,
            font=("Arial", 10)
        )
        self.cb_model.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        self.hide_traj_var = tk.BooleanVar(value=False)
        self.cb_traj = tk.Checkbutton(
            self.root, text="隐藏喷涂加工轨迹", variable=self.hide_traj_var, command=self.toggle_traj,
            font=("Arial", 10)
        )
        self.cb_traj.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=2)

    def toggle_model(self):
        self.visualizer.hide_model = self.hide_model_var.get()
        self.visualizer.apply_visibility()

    def toggle_traj(self):
        self.visualizer.hide_traj = self.hide_traj_var.get()
        self.visualizer.apply_visibility()

    def _read_control_values(self) -> tuple[float, float, float] | None:
        try:
            spacing = float(self.spacing_var.get())
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError
            speed = float(self.speed_var.get())
            if not np.isfinite(speed) or speed <= 0:
                raise ValueError
            spray_distance = float(self.spray_distance_var.get())
            if not np.isfinite(spray_distance):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "目标间距 d 和轨迹速度必须大于 0，喷涂距离必须是有效数字。")
            return None
        return spacing, speed, spray_distance

    def _store_process_values(self, spacing: float, speed: float, spray_distance: float) -> None:
        self.visualizer.current_spacing = spacing
        self.visualizer.trajectory_speed = speed
        self.visualizer.spray_distance = spray_distance

    def update_spacing(self):
        try:
            new_d = float(self.spacing_var.get())
            speed = float(self.speed_var.get())
            spray_distance = float(self.spray_distance_var.get())

            # 【新增】：读取加速度和加加速度
            a_max = float(self.a_max_var.get())
            j_max = float(self.j_max_var.get())

            if not np.isfinite(new_d) or new_d <= 0: raise ValueError
            if not np.isfinite(speed) or speed <= 0: raise ValueError
            if not np.isfinite(a_max) or a_max <= 0: raise ValueError
            if not np.isfinite(j_max) or j_max <= 0: raise ValueError

        except ValueError:
            messagebox.showerror("错误", "参数必须为大于 0 的有效数字。")
            return

        # 存储到 visualizer
        self.visualizer.current_spacing = new_d
        self.visualizer.trajectory_speed = speed
        self.visualizer.spray_distance = spray_distance
        self.visualizer.a_max = a_max  # 【新增】
        self.visualizer.j_max = j_max  # 【新增】

        raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
            self.visualizer.data,
            self.visualizer.precut_seam,
            target_spacing=new_d,
            sample_step=self.visualizer.sample_step
        )
        optimized = optimize_trajectory_arcs_postprocess(
            raw_trajectory,
            self.visualizer.mesh,
            target_spacing=new_d,
            sample_step=self.visualizer.sample_step
        )
        # 【修改5】：交互更新也走 CENTER 能量优化（迭代数可略少，保证响应速度）
        # 搜索 update_spacing 方法中的这行：
        self.visualizer.reset_energy_history()

        # def refresh_iteration(preview, iteration, total_energy, energy_components):
        #     self.visualizer.record_energy(iteration, total_energy)
        #     # 1. 更新 VTK 中的 3D/2D 轨迹数据
        #     self.visualizer.update_trajectory(
        #         preview,
        #         planning_frame,
        #         metadata,
        #         trajectory_2d=raw_trajectory,
        #     )
        #
        #     # 2. 强制 VTK 窗口立即重绘当前帧
        #     self.visualizer.window.Render()
        #
        #     # 3. 强制 Tkinter 立即刷新 GUI 事件队列（必须用 update()，不能用 update_idletasks()）
        #     self.root.update()
        #
        #     cv = float(energy_components.get("uniform_metric", float("nan")))
        #     cv_smooth = float(energy_components.get("uniform_metric_smoothed", float("nan")))
        #     cv_delta = float(energy_components.get("uniform_delta", float("nan")))
        #     cv_rel_delta = float(energy_components.get("uniform_relative_delta", float("nan")))
        #     cv_rel_limit = float(energy_components.get("uniform_relative_tol", float("nan")))
        #     cv_limit = float(energy_components.get("uniform_threshold", float("nan")))
        #     stable_count = int(energy_components.get("uniform_stable_count", 0))
        #     stable_needed = int(energy_components.get("uniform_patience", 0))
        #     print(
        #         f"\r中心轨迹优化中: {iteration}/1  "#500
        #         f"E={total_energy:.6e}  "
        #         f"cov={energy_components['coverage']:.3e}  "
        #         f"spacing={energy_components['spacing']:.3e}  "
        #         f"boundary={energy_components['boundary']:.3e}  "
        #         f"smooth={energy_components['smooth']:.3e}  "
        #         f"CV={cv:.6e}  EMA_CV={cv_smooth:.6e}  "
        #         f"dCV={cv_delta:.3e}  limit={cv_limit:.3e}  "
        #         f"rel_dCV={cv_rel_delta:.3e}  rel_limit={cv_rel_limit:.3e}  "
        #         f"stable={stable_count}/{stable_needed}",
        #         flush=True,
        #     )
        #
        # optimized = optimize_center_trajectories_dual_blank_energy(
        #     optimized,
        #     self.visualizer.mesh,
        #     iterations=1,#500
        #     attract_weight=1.20,
        #     elastic_weight=0.08,
        #     influence_radius=6.0 * new_d,
        #     anneal=0.97,
        #     target_spacing=new_d,
        #     sample_step=self.visualizer.sample_step,
        #     on_iteration=refresh_iteration,
        # )
        self.visualizer.update_trajectory(
            optimized, planning_frame, metadata, trajectory_2d=raw_trajectory
        )
        print(f"成功将目标间距 d 重新规划并更新为: {new_d:.6g}")

    def export_trajectory(self):
        values = self._read_control_values()
        if values is None:
            return
        spacing, speed, spray_distance = values
        self._store_process_values(spacing, speed, spray_distance)

        if not self.visualizer.current_trajectory or self.visualizer.current_metadata is None:
            messagebox.showerror("错误", "当前没有可导出的轨迹，请先更新轨迹。")
            return

        _export_trajectory(
            self.visualizer.csv_path,
            self.visualizer.metadata_path,
            self.visualizer.current_trajectory,
            self.visualizer.current_metadata,
        )
        _export_paq_trajectory(
            self.visualizer.paq_path,
            self.visualizer.mesh,
            self.visualizer.current_trajectory,
            self.visualizer.trajectory_speed,
            self.visualizer.spray_distance,
        )
        messagebox.showinfo("完成", f"轨迹已导出:\n{self.visualizer.paq_path}")
        print(f"Exported PAQ trajectory: {self.visualizer.paq_path}")


def tick_tkinter(obj, event, tk_panel: TkControlPanel):
    try:
        tk_panel.root.update()
    except tk.TclError:
        pass


def _show_result(
        mesh: trimesh.Trimesh,
        data: ObjUVData,
        precut_seam: PrecutSeam,
        cut_edges: np.ndarray,
        trajectory: Sequence[TrajectorySegment],
        planning_frame: PlanningFrame,
        metadata: dict[str, object],
        initial_spacing: float,
        sample_step: float,
        csv_path: Path,
        metadata_path: Path,
        paq_path: Path,
        trajectory_speed: float,
        spray_distance: float,
        trajectory_2d: Sequence[TrajectorySegment] | None = None,
) -> None:
    """Run interactive visualizer without background center-energy optimization."""
    visualizer = InteractiveVisualizer(
        mesh, data, precut_seam, cut_edges, sample_step, initial_spacing,
        csv_path, metadata_path, paq_path, trajectory_speed, spray_distance
    )
    # 绘制传入的超椭圆拟合后轨迹
    visualizer.update_trajectory(trajectory, planning_frame, metadata, trajectory_2d=trajectory_2d)
    visualizer.reset_cameras()
    panel = TkControlPanel(visualizer)

    visualizer.interactor.AddObserver("TimerEvent", lambda obj, ev: tick_tkinter(obj, ev, panel))
    visualizer.interactor.CreateRepeatingTimer(10)

    # 1. 先渲染出 VTK 窗口
    visualizer.window.Render()
    visualizer.interactor.Initialize()

    # # 2. 窗口打开后，实时进行 CENTER 双空白能量场实时优化渲染
    # print("\n[开始实时能量场轨迹优化...]")

    # visualizer.reset_energy_history()
    #
    # def live_iteration_callback(preview, iteration, total_energy, energy_components):
    #     visualizer.record_energy(iteration, total_energy)
    #     visualizer.update_trajectory(
    #         preview, planning_frame, metadata, trajectory_2d=trajectory_2d
    #     )
    #     visualizer.window.Render()
    #     panel.root.update()
    #     cv = float(energy_components.get("uniform_metric", float("nan")))
    #     cv_smooth = float(energy_components.get("uniform_metric_smoothed", float("nan")))
    #     cv_delta = float(energy_components.get("uniform_delta", float("nan")))
    #     cv_rel_delta = float(energy_components.get("uniform_relative_delta", float("nan")))
    #     cv_rel_limit = float(energy_components.get("uniform_relative_tol", float("nan")))
    #     cv_limit = float(energy_components.get("uniform_threshold", float("nan")))
    #     stable_count = int(energy_components.get("uniform_stable_count", 0))
    #     stable_needed = int(energy_components.get("uniform_patience", 0))
    #     print(
    #         f"\r实时优化进度: {iteration}/1  "  # 500
    #         f"E={total_energy:.6e}  "
    #         f"cov={energy_components['coverage']:.3e}  "
    #         f"spacing={energy_components['spacing']:.3e}  "
    #         f"boundary={energy_components['boundary']:.3e}  "
    #         f"smooth={energy_components['smooth']:.3e}  "
    #         f"CV={cv:.6e}  EMA_CV={cv_smooth:.6e}  "
    #         f"dCV={cv_delta:.3e}  limit={cv_limit:.3e}  "
    #         f"rel_dCV={cv_rel_delta:.3e}  rel_limit={cv_rel_limit:.3e}  "
    #         f"stable={stable_count}/{stable_needed}",
    #         flush=True,
    #     )

    # optimized_trajectory = optimize_center_trajectories_dual_blank_energy(
    #     trajectory=list(trajectory),
    #     mesh=mesh,
    #     iterations=1,  # 500
    #     attract_weight=1.20,
    #     elastic_weight=0.08,
    #     influence_radius=6.0 * initial_spacing,
    #     anneal=0.97,
    #     target_spacing=initial_spacing,
    #     sample_step=sample_step,
    #     on_iteration=live_iteration_callback,
    # )
    visualizer.current_trajectory = list(trajectory)
    print("\n[轨迹准备就绪，开启可视化界面！]")
    visualizer.current_trajectory = list(trajectory)

    # 进入主事件循环
    visualizer.interactor.Start()


def _request_spacing(
        mesh: trimesh.Trimesh,
        data: ObjUVData,
        seam: PrecutSeam,
        cut_edges: np.ndarray,
        initial_spacing: float,
        initial_a_max: float,  # 【新增】
        initial_j_max: float,  # 【新增】
) -> tuple[float | None, float, float]:  # 【修改返回值类型注解】
    from vtktrajdisplay import request_spacing
    frame = _planning_frame(_resolve_main_seam_uv_segments(data, seam))

    # 【修改】：传入 a_max 和 j_max，并接收三元组返回值
    result = request_spacing(
        mesh, data.uv, data.uv_faces, initial_spacing,
        cut_edges=cut_edges, planning_frame=frame,
        initial_a_max=initial_a_max, initial_j_max=initial_j_max,  # 【新增参数】
    )

    if result is None:
        return None, None, None

    spacing, a_max, j_max = result
    return spacing, a_max, j_max


def _play_iteration_animation(mesh: trimesh.Trimesh, result_obj: Path) -> None:
    from vtktrajdisplay import play_iteration_animation

    play_iteration_animation(
        mesh, result_obj, _read_iteration_obj, FINAL_FRAME_HOLD_MS,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OptCuts and generate a seam-aware burner coverage trajectory."
    )
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY/OFF; opens a file dialog when omitted.")
    parser.add_argument("--spacing", type=float, default=DEFAULT_TRAJECTORY_SPACING,
                        help=f"Requested track spacing d (default: {DEFAULT_TRAJECTORY_SPACING}).")
    parser.add_argument("--sample-step", type=float, default=0.0,
                        help="3D trajectory sampling step; default is d/4.")
    parser.add_argument("--trajectory-out", default="", help="Output CSV path.")
    parser.add_argument("--metadata-out", default="", help="Output planning JSON path.")
    parser.add_argument("--paq-out", default="", help="Output PAQ-style TXT trajectory path.")
    parser.add_argument("--trajectory-speed", type=float, default=DEFAULT_TRAJECTORY_SPEED,
                        help=f"PAQ trajectory speed (default: {DEFAULT_TRAJECTORY_SPEED}).")
    parser.add_argument("--spray-distance", type=float, default=DEFAULT_SPRAY_DISTANCE,
                        help=f"Inward normal offset distance for PAQ export (default: {DEFAULT_SPRAY_DISTANCE}).")

    # 【新增】：最大加速度和最大加加速度参数
    parser.add_argument("--a-max", type=float, default=1000.0,
                        help="Robot maximum acceleration in mm/s² (default: 1000.0).")
    parser.add_argument("--j-max", type=float, default=5000.0,
                        help="Robot maximum jerk in mm/s³ (default: 5000.0).")

    parser.add_argument("--no-animation", action="store_true", help="Skip the OptCuts iteration animation.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the final VTK/OpenGL window.")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    if args.trajectory_speed <= 0.0 or not np.isfinite(args.trajectory_speed):
        raise ValueError("Trajectory speed must be greater than zero.")
    if not np.isfinite(args.spray_distance):
        raise ValueError("Spray distance must be finite.")
    selected = args.mesh or _select_mesh_file()
    if not selected:
        return
    mesh = _load_mesh(selected)
    optcuts_mesh, precut_seam = _precut_two_boundaries(mesh)
    if precut_seam is None:
        raise ValueError(
            "Seam-aware trajectory planning requires the original burner mesh to have exactly two openings."
        )
    official_input = _prepare_official_input(optcuts_mesh)
    _save_precut_seam(official_input, precut_seam)
    result_obj = _run_official_optcuts(official_input)
    data = _read_obj_uv_data(result_obj)
    optimized_cut_edges = _read_obj_cut_edges(result_obj)
    if len(precut_seam.edges) and len(optimized_cut_edges):
        cut_edges = np.vstack([precut_seam.edges, optimized_cut_edges])
    elif len(precut_seam.edges):
        cut_edges = precut_seam.edges
    else:
        cut_edges = optimized_cut_edges
    if args.no_show:
        selected_spacing = args.spacing
        selected_a_max = args.a_max
        selected_j_max = args.j_max
    else:
        # 【修改】：调用 _request_spacing 时传入 a_max 和 j_max
        selected_spacing, selected_a_max, selected_j_max = _request_spacing(
            mesh, data, precut_seam, cut_edges, args.spacing,
            initial_a_max=args.a_max,  # 【新增】
            initial_j_max=args.j_max,  # 【新增】
        )

    if selected_spacing is None:
        print("Trajectory generation cancelled.")
        return
    sample_step_val = selected_spacing / 4.0 if args.sample_step <= 0.0 else args.sample_step
    raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
        data, precut_seam, target_spacing=selected_spacing, sample_step=sample_step_val
    )

    # 【修改】：在调用优化函数时，传入选定的 a_max 和 j_max
    # 找到原有这句：
    # arc_trajectory = optimize_trajectory_arcs_postprocess(...)

    # 替换为：
    arc_trajectory = optimize_trajectory_arcs_postprocess(
        trajectory=raw_trajectory,
        mesh=mesh,
        target_spacing=selected_spacing,
        sample_step=sample_step_val,
    )

    # 【注意】：不要在这里直接调用 optimize_center_trajectories_dual_blank_energy！
    # 把它移入 _show_result 中，窗口打开后再实时边计算边渲染。

    selected_path = Path(selected)
    csv_path = Path(args.trajectory_out) if args.trajectory_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.csv"
    )
    metadata_path = Path(args.metadata_out) if args.metadata_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.json"
    )
    paq_path = Path(args.paq_out) if args.paq_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory_PAQ.txt"
    )

    print(f"Pre-cut shortest-boundary seam edges: {len(precut_seam.edges)}")
    print(f"Additional detected OptCuts seam edges: {len(optimized_cut_edges)}")
    print(f"Trajectory events: {len(arc_trajectory)}")

    if not args.no_show:
        _show_result(
            mesh, data, precut_seam, cut_edges, arc_trajectory, planning_frame, metadata,
            selected_spacing, sample_step_val, csv_path, metadata_path,
            paq_path, args.trajectory_speed, args.spray_distance,
            trajectory_2d=raw_trajectory,
        )
    if not args.no_animation:
        _play_iteration_animation(optcuts_mesh, result_obj)


if __name__ == "__main__":
    main()
