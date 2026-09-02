#带优化的自适应优化、目前最优
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import queue
import subprocess
import threading
import time
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
from scipy.interpolate import make_lsq_spline
#截断变圆弧
# This script lives one directory deeper than the other STL entrypoints:
# .../PCLPrecess/python/stl/备份/linshi.py -> PCLPrecess.
_SCRIPT_PATH = Path(__file__).resolve()
ROOT = (_SCRIPT_PATH.parents[3]
        if _SCRIPT_PATH.parent.name == "备份"
        else _SCRIPT_PATH.parents[2])
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
        title="Select mesh or OptCuts cache",
        filetypes=[
            ("OptCuts cache", "*.json"),
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


def _cutopt_cache_path(mesh_path: Path) -> Path:
    return mesh_path.with_name(mesh_path.stem + "_cutopt.json")


def _save_cutopt_cache(mesh_path: Path, official_input: Path, result_obj: Path,
                       seam: PrecutSeam, cut_edges: np.ndarray) -> Path:
    cache = _cutopt_cache_path(mesh_path)
    result_copy = mesh_path.with_name(mesh_path.stem + "_cutopt_result.obj")
    shutil.copy2(result_obj, result_copy)
    payload = {
        "format": "optcuts-cache-v1",
        "model": {"path": str(mesh_path.resolve()), "name": mesh_path.name,
                  "size": mesh_path.stat().st_size, "mtime_ns": mesh_path.stat().st_mtime_ns},
        "optcuts": {"unfolded_obj": str(result_copy.resolve())},
    }
    cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved OptCuts cache: {cache}")
    return cache


def _load_cutopt_cache(cache: Path):
    payload = json.loads(cache.read_text(encoding="utf-8"))
    if payload.get("format") != "optcuts-cache-v1":
        raise ValueError("Unsupported OptCuts cache format.")
    model_path = Path(payload["model"]["path"])
    result_obj = Path(payload["optcuts"]["unfolded_obj"])
    for path in (model_path, result_obj):
        if not path.exists():
            raise FileNotFoundError(f"Cached file not found: {path}")
    mesh = _load_mesh(str(model_path))
    optcuts_mesh, seam = _precut_two_boundaries(mesh)
    return (model_path, optcuts_mesh, result_obj, seam,
            _read_obj_uv_data(result_obj), _read_obj_cut_edges(result_obj))


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


def _logical_surface_tracks(trajectory: Sequence[TrajectorySegment]) -> list[dict[str, object]]:
    """Return individual scan polylines, including halves merged by phases."""
    tracks: list[dict[str, object]] = []
    for segment_index, segment in enumerate(trajectory):
        if segment.kind != "SURFACE_SCAN" or len(segment.xyz) < 2:
            continue
        region = str(getattr(segment, "region", "")).upper()
        phases = getattr(segment, "phases", None)
        if region in {"LOWER", "UPPER"} and phases is not None:
            phase_values = np.asarray(phases)
            for phase in (0, 1):
                indices = np.flatnonzero(phase_values == phase)
                if len(indices) >= 2:
                    tracks.append({
                        "xyz": np.asarray(segment.xyz[indices], dtype=float),
                        "uv": np.asarray(segment.uv[indices], dtype=float),
                        "region": region,
                        "segment_index": segment_index,
                        "phase": phase,
                    })
            continue
        tracks.append({
            "xyz": np.asarray(segment.xyz, dtype=float),
            "uv": np.asarray(segment.uv, dtype=float),
            "region": region,
            "segment_index": segment_index,
            "phase": None,
        })
    return tracks


def _segment_distance_3d(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest distance between two finite 3D segments."""
    p0, p1 = np.asarray(first, float)
    q0, q1 = np.asarray(second, float)
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denominator = a * c - b * b
    s_num, s_den = denominator, denominator
    t_num, t_den = denominator, denominator
    if denominator <= EPS:
        s_num, s_den, t_num, t_den = 0.0, 1.0, e, c
    else:
        s_num = b * e - c * d
        t_num = a * e - b * d
        if s_num < 0.0:
            s_num, t_num, t_den = 0.0, e, c
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, e + b, c
    if t_num < 0.0:
        t_num = 0.0
        if -d < 0.0:
            s_num = 0.0
        elif -d > a:
            s_num = s_den
        else:
            s_num, s_den = -d, a
    elif t_num > t_den:
        t_num = t_den
        if -d + b < 0.0:
            s_num = 0.0
        elif -d + b > a:
            s_num = s_den
        else:
            s_num, s_den = -d + b, a
    s = 0.0 if abs(s_num) <= EPS else s_num / max(s_den, EPS)
    t = 0.0 if abs(t_num) <= EPS else t_num / max(t_den, EPS)
    return float(np.linalg.norm(w + s * u - t * v))


def _line_to_points_distance(line: np.ndarray, points: np.ndarray) -> float:
    samples = np.linspace(np.asarray(line[0], float), np.asarray(line[1], float), 17)
    return float(np.min(cKDTree(np.asarray(points, float)).query(samples)[0]))


def _track_slice(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Follow a sampled track from the point nearest start to the point nearest end."""
    points = np.asarray(points, float)
    tree = cKDTree(points)
    start_idx = int(tree.query(np.asarray(start, float))[1])
    end_idx = int(tree.query(np.asarray(end, float))[1])
    if start_idx <= end_idx:
        middle = points[start_idx:end_idx + 1]
    else:
        middle = points[start_idx::-1] if end_idx == 0 else points[start_idx:end_idx - 1:-1]
    return np.vstack([np.asarray(start, float), middle, np.asarray(end, float)])


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd test for a 2D polygon."""
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        crosses = ((py[current] > y) != (py[previous] > y)) & (
            x < (px[previous] - px[current]) * (y - py[current])
            / (py[previous] - py[current] + EPS) + px[current]
        )
        inside ^= crosses
        previous = current
    return inside


def _local_connected_face_mask(
    mesh: trimesh.Trimesh,
    candidate_mask: np.ndarray,
    boundary_points: np.ndarray,
) -> np.ndarray:
    """Keep the candidate face component attached to the local boundary."""
    candidate_ids = np.flatnonzero(candidate_mask)
    if not len(candidate_ids):
        return candidate_mask
    triangle_centers = np.asarray(mesh.triangles_center, dtype=float)
    boundary_tree = cKDTree(np.asarray(boundary_points, dtype=float))
    seed = int(candidate_ids[np.argmin(
        boundary_tree.query(triangle_centers[candidate_ids])[0]
    )])

    allowed = np.asarray(candidate_mask, dtype=bool)
    graph: dict[int, list[int]] = {}
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency = adjacency[np.all(allowed[adjacency], axis=1)]
    for first, second in adjacency:
        first, second = int(first), int(second)
        graph.setdefault(first, []).append(second)
        graph.setdefault(second, []).append(first)

    component = np.zeros(len(candidate_mask), dtype=bool)
    component[seed] = True
    stack = [seed]
    while stack:
        face = stack.pop()
        for neighbour in graph.get(face, ()):
            if not component[neighbour]:
                component[neighbour] = True
                stack.append(neighbour)
    return component


def _project_to_mesh(mesh: trimesh.Trimesh | None, point: np.ndarray) -> np.ndarray | None:
    if mesh is None or point is None or not np.isfinite(point).all():
        return None
    try:
        projected, _, _ = trimesh.proximity.closest_point(mesh, np.asarray(point, float)[None, :])
        return np.asarray(projected[0], float)
    except Exception:
        try:
            projected, _, _ = trimesh.proximity.closest_point_naive(
                mesh, np.asarray(point, float)[None, :]
            )
            return np.asarray(projected[0], float)
        except Exception:
            vertices = np.asarray(mesh.vertices, float)
            return vertices[int(cKDTree(vertices).query(np.asarray(point, float))[1])] if len(vertices) else None


def _closed_polyline_center(points: np.ndarray) -> np.ndarray:
    """Geometric centroid of a sampled 3D curve closed by its endpoint chord."""
    points = np.asarray(points, float)
    return np.mean(points, axis=0)


def _radial_barrier_state(
    points: np.ndarray,
    barriers: Sequence[dict[str, object]],
    spacing: float,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the strongest radial hard barrier at each point."""
    points = np.asarray(points, float)
    count = len(points)
    force = np.zeros_like(points)
    direction = np.zeros_like(points)
    magnitude = np.zeros(count, dtype=float)
    dangerous = np.zeros(count, dtype=bool)
    inside_any = np.zeros(count, dtype=bool)
    escape_step = np.zeros(count, dtype=float)
    d = max(float(spacing), EPS)
    inner_limit = 0.8 * d
    outer_limit = 1.2 * d

    for barrier in barriers:
        center = np.asarray(barrier["center"], float)
        tree = barrier["tree"]
        distance, nearest_index = tree.query(points)
        nearest_index = np.asarray(nearest_index, dtype=np.intp)
        tree_points = np.asarray(tree.data, dtype=float)
        nearest = tree_points[nearest_index]
        ray = points - center
        radius = np.linalg.norm(ray, axis=1)
        valid = radius > 1.0e-8
        unit = np.zeros_like(points)
        unit[valid] = ray[valid] / radius[valid, None]
        boundary_radius = np.linalg.norm(nearest - center, axis=1)
        at_center = ~valid & (boundary_radius > 1.0e-8)
        unit[at_center] = (
            (nearest[at_center] - center)
            / boundary_radius[at_center, None]
        )
        inside = radius < boundary_radius - 1.0e-6 * d
        ultra_close = (~inside) & (distance < inner_limit)
        transition = (~inside) & (~ultra_close) & (distance < outer_limit)

        local_magnitude = np.zeros(count, dtype=float)
        depth = np.maximum(boundary_radius - radius, 0.0) / d
        local_magnitude[inside] = stiffness * (
            20.0 + 20.0 * depth[inside] + 10.0 * depth[inside] ** 2
        )
        close_ratio = np.clip((inner_limit - distance) / inner_limit, 0.0, 1.0)
        local_magnitude[ultra_close] = stiffness * (
            3.0 + 17.0 * close_ratio[ultra_close] ** 2
        )
        transition_t = np.clip(
            (outer_limit - distance) / max(outer_limit - inner_limit, EPS),
            0.0,
            1.0,
        )
        transition_smooth = transition_t ** 2 * (3.0 - 2.0 * transition_t)
        local_magnitude[transition] = stiffness * 3.0 * transition_smooth[transition]

        stronger = local_magnitude > magnitude
        if np.any(stronger):
            magnitude[stronger] = local_magnitude[stronger]
            direction[stronger] = unit[stronger]
        dangerous |= inside | ultra_close
        inside_any |= inside

        needed = np.where(
            inside,
            np.maximum(boundary_radius - radius, 0.0) + 0.85 * d,
            np.maximum(0.85 * d - distance, 0.0),
        )
        local_escape = np.where(
            inside | ultra_close,
            np.clip(needed, 0.6 * d, 0.8 * d),
            0.0,
        )
        escape_step = np.maximum(escape_step, local_escape)

    force = direction * magnitude[:, None]
    return force, direction, dangerous, inside_any, escape_step


def _special_boundary_geometry(
    trajectory: Sequence[TrajectorySegment],
    spacing: float,
    mesh: trimesh.Trimesh | None = None,
) -> dict[str, object]:
    """Find periodic center-end neighbours and blue boundary connectors.

    - The first (center[0]) and last (center[-1]) CENTER tracks are adjacent in physical space.
    - When UPPER / LOWER regions exist, center[0] and center[-1] independently search for their
      nearest outer track branches at 1.5d distance.
    - If UPPER or LOWER is absent on one (or both) sides, the boundary endpoints of center[0]
      and center[-1] on that side are directly connected.
    """
    if "_logical_surface_tracks" in globals():
        tracks = _logical_surface_tracks(trajectory)
    else:
        tracks = []
        for seg_idx, seg in enumerate(trajectory):
            region = str(getattr(seg, "region", "")).upper()
            if seg.kind != "SURFACE_SCAN" or region not in ("CENTER", "LOWER", "UPPER") or len(seg.xyz) < 2:
                continue
            phases = getattr(seg, "phases", None)
            if region in ("LOWER", "UPPER") and phases is not None:
                values = np.asarray(phases)
                for phase in (0, 1):
                    ids = np.flatnonzero(values == phase)
                    if len(ids) >= 2:
                        tracks.append({
                            "region": region,
                            "segment_index": seg_idx,
                            "phase": phase,
                            "xyz": np.asarray(seg.xyz[ids], float),
                            "uv": np.asarray(seg.uv[ids], float) if len(seg.uv) == len(seg.xyz) else np.full((len(ids), 2), np.nan),
                        })
            else:
                tracks.append({
                    "region": region,
                    "segment_index": seg_idx,
                    "phase": 0,
                    "xyz": np.asarray(seg.xyz, float),
                    "uv": np.asarray(seg.uv, float) if len(seg.uv) == len(seg.xyz) else np.full((len(seg.xyz), 2), np.nan),
                })

    center = [t for t in tracks if t["region"] == "CENTER"]
    target = 1.5 * float(spacing)
    if not center:
        return {
            "tracks": [],
            "connectors_3d": [],
            "connectors_2d": [],
            "connector_records": [],
            "selected_void_connectors_3d": {},
            "blank_centers_3d": {},
            "repulsion_centers_3d": {},
            "void_center_eligibility": {},
            "periodic_pair_blocked": False,
            "target_distance": target,
        }

    c_first = center[0]
    c_last = center[-1]
    special_tracks: list[dict[str, object]] = [c_first, c_last] if len(center) > 1 else [c_first]

    # 1. 基础匹配：center 内部首尾闭环 (C_first <-> C_last)
    pairs: list[tuple[dict[str, object], dict[str, object], str, str | None]] = [
        (c_first, c_last, "PERIODIC", None),
    ]

    # 2. 跨区独立匹配：为 C_first 和 C_last 分别独立寻找最近的分支
    has_lower = any(t["region"] == "LOWER" for t in tracks)
    has_upper = any(t["region"] == "UPPER" for t in tracks)

    for region in ("LOWER", "UPPER"):
        candidates = [t for t in tracks if t["region"] == region]
        if not candidates:
            continue

        cand_first = min(
            candidates,
            key=lambda t: float(np.min(cKDTree(np.asarray(t["xyz"], float)).query(np.asarray(c_first["xyz"], float))[0]))
        )
        cand_last = min(
            candidates,
            key=lambda t: float(np.min(cKDTree(np.asarray(t["xyz"], float)).query(np.asarray(c_last["xyz"], float))[0]))
        )

        # Do not use ``dict in list`` here: NumPy point arrays make dictionary
        # equality attempt broadcasting when shapes differ.  The normalized
        # track key deduplication below handles repeated branches safely.
        special_tracks.append(cand_first)
        special_tracks.append(cand_last)

        pairs.append((c_first, cand_first, "FIRST", region))
        pairs.append((c_last, cand_last, "LAST", region))

    connectors_3d: list[list[list[float]]] = []
    connectors_2d: list[list[list[float]]] = []
    connector_records: list[dict[str, object]] = []

    # 3. 求解所有匹配对的 1.5d 等距截断点
    for centre_track, outer_track, centre_role, boundary_region in pairs:
        c_xyz = np.asarray(centre_track["xyz"], float)
        o_xyz = np.asarray(outer_track["xyz"], float)
        c_uv = np.asarray(centre_track["uv"], float)
        o_uv = np.asarray(outer_track["uv"], float)

        distances, _ = cKDTree(o_xyz).query(c_xyz)
        values = distances - target
        roots: list[tuple[np.ndarray, np.ndarray, int, float]] = []

        if abs(values[0]) <= 1.0e-9:
            _, o_idx = cKDTree(o_xyz).query(c_xyz[0])
            roots.append((c_xyz[0], o_xyz[int(o_idx)], 0, 0.0))

        for index in range(len(values) - 1):
            if values[index] == 0.0:
                ratio = 0.0
            elif values[index] * values[index + 1] < 0.0:
                ratio = float(np.clip(-values[index] / (values[index + 1] - values[index]), 0.0, 1.0))
            else:
                continue
            point = c_xyz[index] + ratio * (c_xyz[index + 1] - c_xyz[index])
            _, o_idx = cKDTree(o_xyz).query(point)
            roots.append((point, o_xyz[int(o_idx)], index, ratio))

        if abs(values[-1]) <= 1.0e-9:
            _, o_idx = cKDTree(o_xyz).query(c_xyz[-1])
            roots.append((c_xyz[-1], o_xyz[int(o_idx)], len(values) - 2, 1.0))

        for point, o_point, index, ratio in roots:
            line_3d = [point.tolist(), o_point.tolist()]
            connectors_3d.append(line_3d)
            line_2d = None
            if len(c_uv) == len(c_xyz) and len(o_uv):
                uv_point = c_uv[index] + ratio * (c_uv[index + 1] - c_uv[index])
                _, o_uv_idx = cKDTree(o_xyz).query(o_point)
                line_2d = [uv_point.tolist(), o_uv[int(o_uv_idx)].tolist()]
                connectors_2d.append(line_2d)
            connector_records.append({
                "kind": "PERIODIC" if centre_role == "PERIODIC" else "BOUNDARY",
                "side": boundary_region,
                "center_role": centre_role,
                "line_3d": line_3d,
                "line_2d": line_2d,
                "outer_segment_index": outer_track.get("segment_index"),
                "outer_phase": outer_track.get("phase"),
            })

    # 4. 边界端点直连：若某侧无 UPPER 或 LOWER，直接连接该侧在物理边界上的两个端点
    if len(center) >= 2 and (not has_lower or not has_upper):
        c_f_xyz = np.asarray(c_first["xyz"], float)
        c_l_xyz = np.asarray(c_last["xyz"], float)
        c_f_uv = np.asarray(c_first["uv"], float)
        c_l_uv = np.asarray(c_last["uv"], float)

        pts_f_3d = [c_f_xyz[0], c_f_xyz[-1]]
        pts_f_2d = [c_f_uv[0], c_f_uv[-1]] if len(c_f_uv) == len(c_f_xyz) else [None, None]

        pts_l_3d = [c_l_xyz[0], c_l_xyz[-1]]
        pts_l_2d = [c_l_uv[0], c_l_uv[-1]] if len(c_l_uv) == len(c_l_xyz) else [None, None]

        # 欧氏距离对齐同侧端点
        d_direct = np.linalg.norm(pts_f_3d[0] - pts_l_3d[0]) + np.linalg.norm(pts_f_3d[1] - pts_l_3d[1])
        d_cross = np.linalg.norm(pts_f_3d[0] - pts_l_3d[1]) + np.linalg.norm(pts_f_3d[1] - pts_l_3d[0])

        if d_direct <= d_cross:
            end_pairs = [
                {"p1": pts_f_3d[0], "p2": pts_l_3d[0], "uv1": pts_f_2d[0], "uv2": pts_l_2d[0]},
                {"p1": pts_f_3d[1], "p2": pts_l_3d[1], "uv1": pts_f_2d[1], "uv2": pts_l_2d[1]},
            ]
        else:
            end_pairs = [
                {"p1": pts_f_3d[0], "p2": pts_l_3d[1], "uv1": pts_f_2d[0], "uv2": pts_l_2d[1]},
                {"p1": pts_f_3d[1], "p2": pts_l_3d[0], "uv1": pts_f_2d[1], "uv2": pts_l_2d[0]},
            ]

        lower_pts = np.vstack([t["xyz"] for t in tracks if t["region"] == "LOWER"]) if has_lower else None
        upper_pts = np.vstack([t["xyz"] for t in tracks if t["region"] == "UPPER"]) if has_upper else None

        connect_pair_indices = set()
        if has_lower and not has_upper:
            # 离 LOWER 近的为 LOWER 侧，远离 LOWER 的即为缺失的 UPPER 边界
            d0 = float(cKDTree(lower_pts).query(end_pairs[0]["p1"])[0])
            d1 = float(cKDTree(lower_pts).query(end_pairs[1]["p1"])[0])
            upper_side_idx = 1 if d0 <= d1 else 0
            connect_pair_indices.add(upper_side_idx)
        elif has_upper and not has_lower:
            # 离 UPPER 近的为 UPPER 侧，远离 UPPER 的即为缺失的 LOWER 边界
            d0 = float(cKDTree(upper_pts).query(end_pairs[0]["p1"])[0])
            d1 = float(cKDTree(upper_pts).query(end_pairs[1]["p1"])[0])
            lower_side_idx = 1 if d0 <= d1 else 0
            connect_pair_indices.add(lower_side_idx)
        elif not has_lower and not has_upper:
            connect_pair_indices.update([0, 1])

        for p_idx in connect_pair_indices:
            pair_data = end_pairs[p_idx]
            line_3d = [pair_data["p1"].tolist(), pair_data["p2"].tolist()]
            connectors_3d.append(line_3d)
            line_2d = None
            if pair_data["uv1"] is not None and pair_data["uv2"] is not None:
                line_2d = [pair_data["uv1"].tolist(), pair_data["uv2"].tolist()]
                connectors_2d.append(line_2d)
            missing_side = (
                "UPPER" if has_lower and not has_upper else
                "LOWER" if has_upper and not has_lower else
                ("LOWER" if p_idx == 0 else "UPPER")
            )
            connector_records.append({
                "kind": "MODEL_BOUNDARY",
                "side": missing_side,
                "center_role": "ENDPOINTS",
                "line_3d": line_3d,
                "line_2d": line_2d,
                "outer_segment_index": None,
                "outer_phase": None,
            })

    # 去重并输出
    # Keep the hard-boundary choice identical to intrusion handling.
    raw_boundary_tracks: dict[str, np.ndarray] = {}
    for side in ("LOWER", "UPPER"):
        candidates = [
            np.asarray(seg.xyz, float) for seg in trajectory
            if seg.kind == "SURFACE_SCAN"
            and str(getattr(seg, "region", "")).upper() == side
            and len(seg.xyz) >= 2
        ]
        if candidates:
            raw_boundary_tracks[side] = candidates[-1] if side == "LOWER" else candidates[0]

    periodic_records = [r for r in connector_records if r["kind"] == "PERIODIC"]
    direct_by_side = {
        str(r["side"]): r for r in connector_records if r["kind"] == "MODEL_BOUNDARY"
    }
    side_targets: dict[str, np.ndarray] = {}
    for side in ("LOWER", "UPPER"):
        if side in raw_boundary_tracks:
            side_targets[side] = raw_boundary_tracks[side]
        elif side in direct_by_side:
            side_targets[side] = np.asarray(direct_by_side[side]["line_3d"], float)

    periodic_for_side: dict[str, dict[str, object]] = {}
    available_sides = [side for side in ("LOWER", "UPPER") if side in side_targets]
    if len(periodic_records) == 1:
        for side in available_sides:
            periodic_for_side[side] = periodic_records[0]
    elif len(periodic_records) == 2 and len(available_sides) == 2:
        # Exactly two lines: use the requested minimum-total-distance matching.
        _, lower_index, upper_index = min(
            (
                _line_to_points_distance(np.asarray(periodic_records[i]["line_3d"], float), side_targets["LOWER"])
                + _line_to_points_distance(np.asarray(periodic_records[j]["line_3d"], float), side_targets["UPPER"]),
                i,
                j,
            )
            for i in range(len(periodic_records))
            for j in range(len(periodic_records))
            if i != j
        )
        periodic_for_side["LOWER"] = periodic_records[lower_index]
        periodic_for_side["UPPER"] = periodic_records[upper_index]
    elif len(periodic_records) > 2:
        # More than two lines: each side independently takes its nearest line.
        for side in available_sides:
            periodic_for_side[side] = min(
                periodic_records,
                key=lambda record: _line_to_points_distance(
                    np.asarray(record["line_3d"], float), side_targets[side]
                ),
            )
    elif periodic_records:
        for side in available_sides:
            periodic_for_side[side] = min(
                periodic_records,
                key=lambda record: _line_to_points_distance(
                    np.asarray(record["line_3d"], float), side_targets[side]
                ),
            )

    selected_records: dict[str, list[dict[str, object]]] = {}
    selected_void_connectors_3d: dict[str, list[list[list[float]]]] = {}
    blank_centers_3d: dict[str, list[float]] = {}
    repulsion_centers_3d: dict[str, list[float]] = {}
    void_center_eligibility: dict[str, dict[str, object]] = {}
    for side, boundary_track in raw_boundary_tracks.items():
        projected = _project_to_mesh(mesh, _closed_polyline_center(boundary_track))
        if projected is not None:
            repulsion_centers_3d[side] = projected.tolist()

    all_spray_points = [
        np.asarray(seg.xyz, float) for seg in trajectory
        if seg.kind == "SURFACE_SCAN" and len(seg.xyz) >= 2
    ]
    spray_tree = cKDTree(np.vstack(all_spray_points)) if all_spray_points else None
    triangle_centers = (
        np.asarray(mesh.triangles_center, float) if mesh is not None else np.empty((0, 3))
    )

    for side, periodic_record in periodic_for_side.items():
        if side not in raw_boundary_tracks:
            void_center_eligibility[side] = {
                "eligible": False,
                "reason": "missing_boundary_trajectory",
            }
            # A missing red side is still closed by its model-boundary
            # endpoint connector. Keep the selected records for rendering.
            direct_record = direct_by_side.get(side)
            if direct_record is not None:
                selected_records[side] = [periodic_record, direct_record]
                selected_void_connectors_3d[side] = [
                    periodic_record["line_3d"],
                    direct_record["line_3d"],
                ]
            continue
        boundary_tree = cKDTree(raw_boundary_tracks[side])
        first_max_gap = float(np.max(boundary_tree.query(
            np.asarray(c_first["xyz"], float)
        )[0]))
        last_max_gap = float(np.max(boundary_tree.query(
            np.asarray(c_last["xyz"], float)
        )[0]))
        gap_tolerance = 1.0e-9 * max(target, 1.0)
        first_has_void = first_max_gap > target + gap_tolerance
        last_has_void = last_max_gap > target + gap_tolerance
        side_is_eligible = bool(first_has_void and last_has_void)
        void_center_eligibility[side] = {
            "eligible": side_is_eligible,
            "c0_max_gap": first_max_gap,
            "c_last_max_gap": last_max_gap,
            "required_gap": target,
        }
        if not side_is_eligible:
            continue

        periodic_line = np.asarray(periodic_record["line_3d"], float)
        chosen = [periodic_record]
        polygon_parts: list[np.ndarray] = []
        if side in raw_boundary_tracks:
            boundary_candidates = [
                r for r in connector_records
                if r["kind"] == "BOUNDARY" and r["side"] == side
            ]
            first_candidates = [r for r in boundary_candidates if r["center_role"] == "FIRST"]
            last_candidates = [r for r in boundary_candidates if r["center_role"] == "LAST"]
            if not first_candidates or not last_candidates:
                continue
            first_record = min(
                first_candidates,
                key=lambda r: _segment_distance_3d(periodic_line, np.asarray(r["line_3d"], float)),
            )
            last_record = min(
                last_candidates,
                key=lambda r: _segment_distance_3d(periodic_line, np.asarray(r["line_3d"], float)),
            )
            chosen.extend([first_record, last_record])
            first_line = np.asarray(first_record["line_3d"], float)
            last_line = np.asarray(last_record["line_3d"], float)
            polygon_parts = [
                _track_slice(np.asarray(c_first["xyz"], float), periodic_line[0], first_line[0]),
                first_line,
                _track_slice(raw_boundary_tracks[side], first_line[1], last_line[1]),
                last_line[::-1],
                _track_slice(np.asarray(c_last["xyz"], float), last_line[0], periodic_line[1]),
            ]
        elif side in direct_by_side:
            direct_record = direct_by_side[side]
            direct_line = np.asarray(direct_record["line_3d"], float)
            direct_cost = np.linalg.norm(np.asarray(c_first["xyz"], float) - direct_line[0], axis=1).min()
            reverse_cost = np.linalg.norm(np.asarray(c_first["xyz"], float) - direct_line[1], axis=1).min()
            if reverse_cost < direct_cost:
                direct_line = direct_line[::-1]
            chosen.append(direct_record)
            polygon_parts = [
                _track_slice(np.asarray(c_first["xyz"], float), periodic_line[0], direct_line[0]),
                direct_line,
                _track_slice(np.asarray(c_last["xyz"], float), direct_line[1], periodic_line[1]),
            ]

        selected_records[side] = chosen
        selected_void_connectors_3d[side] = [r["line_3d"] for r in chosen]
        if mesh is None or spray_tree is None or not polygon_parts or not len(triangle_centers):
            continue
        polygon_3d = np.vstack([part[:-1] for part in polygon_parts] + [polygon_parts[-1][-1:]])
        if len(polygon_3d) < 3:
            continue
        origin = np.mean(polygon_3d, axis=0)
        _, _, vh = np.linalg.svd(polygon_3d - origin, full_matrices=False)
        basis = vh[:2]
        polygon_2d = (polygon_3d - origin) @ basis.T
        in_region = _points_in_polygon((triangle_centers - origin) @ basis.T, polygon_2d)
        if len(vh) >= 3:
            normal = vh[2]
            boundary_plane_depth = np.abs((polygon_3d - origin) @ normal)
            plane_limit = (
                float(np.percentile(boundary_plane_depth, 95))
                + 0.75 * max(target / 1.5, EPS)
            )
            in_region &= (
                np.abs((triangle_centers - origin) @ normal) <= plane_limit
            )
        in_region = _local_connected_face_mask(mesh, in_region, polygon_3d)
        candidate_ids = np.flatnonzero(in_region)
        if not len(candidate_ids):
            continue
        void_distances = spray_tree.query(triangle_centers[candidate_ids])[0]
        # The face center with the largest clearance is only a discrete seed.
        # Interpolate its strongest neighbouring face-center samples so the
        # reported blank point is not quantized to one triangle.
        order = np.argsort(void_distances)[::-1]
        interpolation_count = min(8, len(order))
        selected = order[:interpolation_count]
        local_distances = np.asarray(void_distances[selected], dtype=float)
        weights = np.maximum(
            local_distances - float(local_distances.min()),
            1.0e-6,
        ) ** 2
        best_center = np.average(
            triangle_centers[candidate_ids[selected]],
            axis=0,
            weights=weights,
        )
        projected = _project_to_mesh(mesh, best_center)
        if projected is not None:
            blank_centers_3d[side] = projected.tolist()

    unique_tracks = []
    seen = set()
    for t in special_tracks:
        key = (t["region"], t.get("segment_index", 0), t.get("phase", 0))
        if key not in seen:
            seen.add(key)
            unique_tracks.append({
                "region": t["region"],
                "segment_index": t.get("segment_index", 0),
                "phase": t.get("phase", 0),
            })

    return {
        "tracks": unique_tracks,
        "connectors_3d": connectors_3d,
        "connectors_2d": connectors_2d,
        "connector_records": connector_records,
        "selected_connector_records": selected_records,
        "selected_void_connectors_3d": selected_void_connectors_3d,
        "blank_centers_3d": blank_centers_3d,
        "repulsion_centers_3d": repulsion_centers_3d,
        "void_center_eligibility": void_center_eligibility,
        # MODEL_BOUNDARY only closes a side that has no red boundary track.
        # It is not evidence that a physical barrier separates C0 and C_last.
        "periodic_pair_blocked": False,
        "target_distance": target,
        "selected_connector_records": selected_records,
    }



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

    # Use a short cubic-like connector whose endpoints are exactly the two
    # surface endpoints.  The previous implementation offset both endpoints
    # by ``exit_direction-entry_direction``.  When the requested overtravel
    # (2*d) was larger than the actual inter-track gap, that produced a large
    # loop which visually detached the green connector from the red scans.
    p0 = np.asarray(previous.xyz[-1], dtype=float)
    p3 = np.asarray(following.xyz[0], dtype=float)
    gap = float(np.linalg.norm(p3 - p0))
    handle = min(max(float(overtravel), 0.0), 0.45 * gap)
    if handle <= EPS:
        xyz = np.vstack([p0, p3])
    else:
        # p1 follows the outgoing tangent; p2 approaches p3 along the
        # incoming tangent.  Both p0 and p3 are retained verbatim.
        p1 = p0 + handle * exit_direction
        p2 = p3 - handle * entry_direction
        xyz = np.vstack([p0, p1, p2, p3])
    return TrajectorySegment(
        kind="OVERTRAVEL",
        uv=np.full((len(xyz), 2), np.nan),
        xyz=xyz,
        spray_on=False,
        note="3D-only opening connector",
        region="TRANSITION"  # 【新增】：标记为过渡段
    )


def _rebuild_overtravel_connectors(
        trajectory: Sequence[TrajectorySegment], overtravel: float,
) -> List[TrajectorySegment]:
    """Recreate connectors after their neighbouring scans have moved."""
    result = list(trajectory)
    for index, segment in enumerate(result):
        if (
                segment.kind == "OVERTRAVEL"
                and index > 0
                and index + 1 < len(result)
                and result[index - 1].kind == "SURFACE_SCAN"
                and result[index + 1].kind == "SURFACE_SCAN"
        ):
            result[index] = _overtravel_segment(
                result[index - 1], result[index + 1], overtravel
            )
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


def _filter_short_boundary_tracks(
        tracks: Sequence[tuple[float, tuple[float, float]]],
        minimum_span: float,
) -> tuple[list[tuple[float, tuple[float, float]]], int]:
    """Drop tiny terminal UV strips produced by a tapered boundary.

    A line merely touching a triangle at the outer boundary is technically a
    valid interval, but it is not a useful spray pass.  Keep the original
    radial order and report how many candidates were removed.
    """
    threshold = max(float(minimum_span), 0.0)
    accepted = [
        track for track in tracks
        if float(track[1][1] - track[1][0]) + 1.0e-9 >= threshold
    ]
    # Never turn a valid side into an empty side solely because of filtering;
    # retain its longest candidate as a conservative fallback.
    if tracks and not accepted:
        accepted = [max(tracks, key=lambda item: item[1][1] - item[1][0])]
    return accepted, len(tracks) - len(accepted)


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


def prune_v_sharp_apex(
        xyz: np.ndarray,
        cutoff_distance: float,
        mesh: trimesh.Trimesh | None = None
) -> np.ndarray:
    """在两臂间距达到 cutoff_distance (1.0*d) 处提前截断，消除切缝死区"""
    if len(xyz) < 8:
        return xyz

    P1, P3 = xyz[0], xyz[-1]
    chord = P3 - P1
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-6:
        return xyz
    u_chord = chord / chord_len
    vecs = xyz - P1
    perp_dists = np.linalg.norm(vecs - np.outer(vecs @ u_chord, u_chord), axis=1)
    tip_idx = int(np.argmax(perp_dists))

    if tip_idx <= 2 or tip_idx >= len(xyz) - 3:
        return xyz

    leg1 = xyz[:tip_idx + 1]
    leg2 = xyz[tip_idx:]

    diffs = np.linalg.norm(leg1[:, None, :] - leg2[None, :, :], axis=2)
    under_cutoff = diffs < cutoff_distance

    if not np.any(under_cutoff):
        return xyz

    i_indices, j_indices = np.where(under_cutoff)
    idx1 = max(0, int(np.min(i_indices)) - 1)
    idx2 = min(len(leg2) - 1, int(np.max(j_indices)) + 1)

    if idx1 >= len(leg1) or idx2 <= 0:
        return xyz

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


def _adaptive_v_cutoff_radius(
        target_spacing: float,
        reference_spacing: float = 10.0,
        reference_ratio: float = 3.0,
        min_ratio: float = 0.5,
        max_ratio: float = 5.0,
        exponent: float = 2.32,
) -> tuple[float, float]:
    """Return a spacing-dependent V-apex cutoff radius and its multiplier.

    The multiplier follows an inverse-power law, then is clipped to a safe
    range.  Thus small track spacing produces a deliberately larger influence
    radius, while large spacing never makes the multiplier vanish.
    """
    d = float(target_spacing)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("target_spacing must be finite and greater than zero")
    if reference_spacing <= 0.0 or reference_ratio <= 0.0:
        raise ValueError("reference spacing and ratio must be positive")
    # Keep a non-zero floor and let only the excess above that floor decay.
    # With the defaults this gives approximately d=10 -> 3, d=20 -> 1,
    # d=40 -> 0.55, while d=5 is clipped to the maximum 5.
    raw_ratio = (
        float(min_ratio)
        + (float(reference_ratio) - float(min_ratio))
        * (float(reference_spacing) / d) ** float(exponent)
    )
    ratio = float(np.clip(raw_ratio, min_ratio, max_ratio))
    return ratio * d, ratio

# =========================================================================
# 辅助函数：超椭圆显式几何值与一阶导数计算
# =========================================================================
def _superellipse_eval_local(x: np.ndarray, a: float, b: float, n: float):
    """
    超椭圆显式方程解析计算：
    y(x) = b * [1 - (|x - a| / a)^n]^(1/n)
    """
    eps_b = 1.0e-5
    u = np.clip(np.abs(x - a) / max(a, 1e-6), 1e-7, 1.0 - eps_b)
    inside = np.maximum(0.0, 1.0 - u ** n)
    y = b * (inside ** (1.0 / n))

    sgn = np.sign(x - a)
    dy = - (b / max(a, 1e-6)) * sgn * (u ** (n - 1.0)) * (inside ** (1.0 / n - 1.0))
    return y, dy


# =========================================================================
# 函数 1：超椭圆参数拟合与解析数据写入
# =========================================================================
def fit_superellipse_trajectories(
        extracted_v_data: List[dict],
        mesh: trimesh.Trimesh,
        target_spacing: float = 30.0,
        sample_step: float = 1.0,
) -> List[TrajectorySegment]:
    cutoff_d, cutoff_ratio = _adaptive_v_cutoff_radius(float(target_spacing))
    print(
        f"Adaptive V cutoff: d={float(target_spacing):.6g}, "
        f"ratio={cutoff_ratio:.6g}, radius={cutoff_d:.6g}"
    )
    effective_step = sample_step if sample_step > 1e-6 else 1.0
    fitted_segments: List[TrajectorySegment] = []
    skipped_short_boundary = 0

    for item in extracted_v_data:
        if item["type"] == "PASSTHROUGH":
            fitted_segments.append(item["seg"])
            continue

        seg = item["seg"]
        xyz_raw = np.asarray(item["xyz"], dtype=float)

        # 1. 提前 d 截断初筛去噪
        xyz_pruned = prune_v_sharp_apex(xyz_raw, cutoff_distance=cutoff_d, mesh=mesh)
        if len(xyz_pruned) < 5:
            # A LOWER/UPPER candidate that collapses to only the two arms and
            # one apex sample is the boundary尖角 itself.  Passing the original
            # polyline through here re-introduces the very sharp first track
            # that the V-apex pruning is meant to remove.  Non-boundary scans
            # are still preserved unchanged.
            if getattr(seg, "region", "") not in ("LOWER", "UPPER"):
                fitted_segments.append(seg)
            else:
                skipped_short_boundary += 1
            continue

        P1 = xyz_pruned[0]
        P3 = xyz_pruned[-1]

        # 2. 建立局部几何坐标系
        M = 0.5 * (P1 + P3)
        X_axis = P3 - P1
        span_2a = np.linalg.norm(X_axis)
        if span_2a < 1e-6:
            fitted_segments.append(seg)
            continue
        X_dir = X_axis / span_2a
        a = span_2a / 2.0

        u_chord = X_dir
        vecs_pruned = xyz_pruned - P1
        perp_dists = np.linalg.norm(vecs_pruned - np.outer(vecs_pruned @ u_chord, u_chord), axis=1)
        idx_tip = int(np.argmax(perp_dists))
        P2 = xyz_pruned[idx_tip]

        height_vec = P2 - M
        h_comp = height_vec - np.dot(height_vec, X_dir) * X_dir
        raw_height = np.linalg.norm(h_comp)
        if raw_height < 1e-6:
            fitted_segments.append(seg)
            continue
        Y_dir = h_comp / raw_height

        # 3. 优化超椭圆参数 (n, b)
        vecs_raw = xyz_pruned - M
        x_raw_local = (vecs_raw @ X_dir) + a
        y_raw_local = vecs_raw @ Y_dir

        min_allowed_b = max(raw_height * 0.3, raw_height - 2.0 * target_spacing)
        max_allowed_b = raw_height

        def fitting_loss(params):
            n_curr, b_curr = params
            y_eval, _ = _superellipse_eval_local(x_raw_local, a, b_curr, n_curr)
            err = np.mean((y_eval - y_raw_local) ** 2)
            mid_mask = np.abs(x_raw_local - a) < 0.6 * a
            mid_err = np.mean((y_eval[mid_mask] - y_raw_local[mid_mask]) ** 2) if np.any(mid_mask) else 0.0
            return err + 1.5 * mid_err

        init_guess = [0.65, raw_height - 0.5 * min(target_spacing, raw_height * 0.1)]
        bounds = [(0.05, 2.0), (min_allowed_b, max_allowed_b)]
        opt_res = minimize(fitting_loss, init_guess, bounds=bounds, method='L-BFGS-B')
        best_n, best_b = opt_res.x

        # 4. 生成离散点与映射
        n_full = max(32, int(np.ceil((2.0 * a) / effective_step)))
        curve_x_loc = np.linspace(0.0, 2.0 * a, n_full)
        curve_y_loc, _ = _superellipse_eval_local(curve_x_loc, a, best_b, best_n)

        curve_3d = M[None, :] + np.outer(curve_x_loc - a, X_dir) + np.outer(curve_y_loc, Y_dir)
        curve_3d[0] = P1
        curve_3d[-1] = P3
        curve_3d, _, _ = trimesh.proximity.closest_point(mesh, curve_3d)
        curve_3d[0] = P1
        curve_3d[-1] = P3

        # 5. 写入工况 A/B 必需的 geometry_meta 字典
        geometry_meta = {
            "type": "SUPERELLIPSE",
            "a": float(a),
            "b": float(best_b),
            "n": float(best_n),
            "raw_height": float(raw_height),
            "P1": P1.tolist(),
            "P3": P3.tolist(),
            "frame_origin": M.tolist(),
            "X_dir": X_dir.tolist(),
            "Y_dir": Y_dir.tolist(),
        }

        if best_n < 0.85:
            shape_type = "三角内摆线"
        elif best_n <= 1.15:
            shape_type = "标准三角"
        elif best_n < 1.95:
            shape_type = "勒洛三角"
        else:
            shape_type = "标准椭圆"

        n_pts = len(curve_3d)
        phases = np.where(np.arange(n_pts) < n_pts // 2, 0, 1)

        fitted_segments.append(TrajectorySegment(
            kind=seg.kind,
            uv=np.full((len(curve_3d), 2), np.nan),
            xyz=curve_3d,
            spray_on=seg.spray_on,
            note=seg.note + f" [{shape_type} n={best_n:.2f}, Raw Superellipse]",
            region=seg.region,
            phases=phases,
            geometry_meta=geometry_meta
        ))

    if skipped_short_boundary:
        print(
            f"[Superellipse skip] {skipped_short_boundary} LOWER/UPPER "
            "tracks collapsed below 5 points after V cutoff."
        )
    return fitted_segments


# =========================================================================
# =========================================================================
# 函数 2：解析内切圆过渡生成（直接以纯外延圆弧替代短小V形与折返线）
# =========================================================================
def apply_tangent_circle_fillet(
        fitted_trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        speed: float,
        a_max: float,
        j_max: float,
        target_spacing: float = 30.0,
        overtravel_factor: float = 2.0,
        sample_step: float = 1.0,
        projection_dist_tol: float = 0.5,
        center_boundary_limit: float | None = None,
        project_fillet_to_mesh: bool = False,
        arc_lock_guard_d: float = 2.0,
) -> List[TrajectorySegment]:
    """
    倒圆与精准投影：
    - 工况 A: 切点在内部，整条轨迹强制无条件贴合投影回曲面；
    - 工况 B: 空间 Rodrigues 圆弧按 a1-a2 弦的模型侧逐点投影；曲面外
      的延伸保留悬空，只有 T1-T2 中点整体位于模型外侧时才完全不投影；
    - 边界熔断: 圆心超出 R + 2d 直接剔除整段；
    - 拓扑闭环: 自动修复 OVERTRAVEL 过渡线。
    """
    if speed <= 0.0 or a_max <= 0.0 or j_max <= 0.0:
        raise ValueError("speed, a_max, and j_max must all be positive")

    R_a = (speed ** 2) / a_max
    R_j = math.sqrt((speed ** 3) / j_max)
    R_min = max(R_a, R_j)

    d = float(target_spacing)
    arc_lock_guard_d = float(arc_lock_guard_d)
    if not np.isfinite(arc_lock_guard_d) or arc_lock_guard_d < 0.0:
        raise ValueError("arc_lock_guard_d must be finite and non-negative.")
    if center_boundary_limit is None or not np.isfinite(center_boundary_limit):
        center_limit = R_min + 2.0 * d
    else:
        center_limit = float(center_boundary_limit)
        if center_limit <= 0.0:
            center_limit = R_min + 2.0 * d
    effective_step = sample_step if sample_step > 1e-6 else 1.0
    case_stats = {"input_superellipse": 0, "passthrough": 0,
                  "case_a": 0, "case_b": 0, "drop_center_limit": 0,
                  "drop_short": 0, "drop_degenerate": 0}

    def project_partial_b_arc(
            points_3d: np.ndarray,
            t1_index: int,
            t2_index: int,
            chord_start: np.ndarray,
            chord_end: np.ndarray,
            model_side_point: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Project only the part of a B arc that can lie on the source face.

        The fitted source track (a1 -> a2) and its apex define the model side
        of the a1-a2 chord.  If the midpoint of T1 -> T2 is on the opposite
        side, the whole circular interval is an air-only transition and must
        remain untouched.  Otherwise this is a boundary/partial case: only
        samples on the model side and close enough to the mesh are projected;
        the remaining samples retain their natural circular extension.
        """
        result = np.asarray(points_3d, dtype=float).copy()
        chord = np.asarray(chord_end, dtype=float) - np.asarray(chord_start, dtype=float)
        chord_len = float(np.linalg.norm(chord))
        if chord_len <= 1e-9 or len(result) == 0:
            return result, False
        chord_unit = chord / chord_len

        # The vector from the a1-a2 line to the original apex points toward
        # the side occupied by the model surface.
        rel_apex = np.asarray(model_side_point, dtype=float) - np.asarray(chord_start, dtype=float)
        apex_lateral = rel_apex - np.dot(rel_apex, chord_unit) * chord_unit
        apex_side_len = float(np.linalg.norm(apex_lateral))
        if apex_side_len <= 1e-9:
            return result, False
        model_side = apex_lateral / apex_side_len

        rel = result - np.asarray(chord_start, dtype=float)
        lateral = rel - (rel @ chord_unit)[:, None] * chord_unit[None, :]
        side_values = lateral @ model_side
        mid_index = int(round(0.5 * (int(t1_index) + int(t2_index))))
        mid_index = min(max(mid_index, 0), len(result) - 1)
        midpoint_side = float(side_values[mid_index])
        side_tol = max(1e-6, 1e-4 * max(chord_len, apex_side_len, d))

        # Entire T1 -> T2 interval is on the air side.  Do not project any
        # point in this case, even if a closest mesh point happens to be near.
        if midpoint_side < -side_tol:
            return result, False

        closest_pts, _, _ = trimesh.proximity.closest_point(mesh, result)
        # Once the midpoint says that this is the surface/partial case, the
        # complete T1->T2 interval is one continuous surface interval.  Using
        # a per-sample distance threshold here creates holes (projected,
        # skipped, projected) when the circular samples are a little farther
        # than the tolerance from a tessellated STL.  Keep the extensions
        # P1->T1 and T2->P3 untouched, and project the continuous middle
        # interval as a whole; it is intentionally not used to punch holes in
        # a surface interval.
        project_mask = np.zeros(len(result), dtype=bool)
        lo = min(max(int(t1_index), 0), len(result) - 1)
        hi = min(max(int(t2_index), lo), len(result) - 1)
        project_mask[lo:hi + 1] = True
        result[project_mask] = closest_pts[project_mask]
        return result, True

    raw_filleted: List[TrajectorySegment] = []

    for i, seg in enumerate(fitted_trajectory):
        if seg.geometry_meta is None or seg.geometry_meta.get("type") not in ("SUPERELLIPSE", "SUPERELLIPSE_FILLET"):
            raw_filleted.append(seg)
            case_stats["passthrough"] += 1
            continue

        case_stats["input_superellipse"] += 1

        meta = seg.geometry_meta
        a = meta["a"]
        b = meta["b"]
        n = meta["n"]
        P1 = np.array(meta["P1"])
        P3 = np.array(meta["P3"])
        M = np.array(meta["frame_origin"])
        X_dir = np.array(meta["X_dir"])
        Y_dir = np.array(meta["Y_dir"])

        # =====================================================================
        # 工况 A：超椭圆解析内切倒圆（工件内表面轨迹，必须无条件投影贴合）
        # =====================================================================
        condition_a_success = False
        # Do not use the old empirical ``R_min <= 0.85*a`` gate.  A is valid
        # exactly when a radius-R_min circle whose centre lies on the V angle
        # bisector has a tangent solution on the fitted superellipse.
        if np.isfinite(R_min) and R_min > 0.0:
            def calc_tangent_dist(x_val):
                y_t, dy_t = _superellipse_eval_local(np.array([x_val]), a, b, n)
                y_v = float(y_t[0])
                dy_v = float(dy_t[0])
                if abs(dy_v) < 1e-6:
                    return 0.0, y_v, b
                Yc_tmp = y_v - (a - x_val) / dy_v
                dist = float(np.hypot(x_val - a, y_v - Yc_tmp))
                return dist, y_v, Yc_tmp

            def objective(x_val):
                dist, _, _ = calc_tangent_dist(x_val)
                return dist - R_min

            x_low = max(a * 0.01, 0.1)
            x_high = a - 1e-3

            try:
                # Search the complete valid half-superellipse instead of
                # assuming the two end samples bracket the root.  The old
                # endpoint-only test rejected valid A solutions when the
                # distance function crossed R_min twice or non-monotonically.
                x_grid = np.linspace(x_low, x_high, 257)
                f_grid = np.asarray([objective(float(x)) for x in x_grid])
                bracket = None
                for j in range(len(x_grid) - 1):
                    f0, f1 = float(f_grid[j]), float(f_grid[j + 1])
                    if not (np.isfinite(f0) and np.isfinite(f1)):
                        continue
                    if abs(f0) <= 1e-8:
                        bracket = (float(x_grid[j]), float(x_grid[j]))
                        break
                    if f0 * f1 < 0.0:
                        # Orient the bracket so the legacy test below remains
                        # meaningful: f_low >= 0 and f_high <= 0.
                        bracket = ((float(x_grid[j]), float(x_grid[j + 1]))
                                   if f0 >= 0.0 else
                                   (float(x_grid[j + 1]), float(x_grid[j])))
                        break
                if bracket is not None:
                    x_low, x_high = bracket
                f_low = objective(x_low)
                f_high = objective(x_high)
                if f_high <= 0 and f_low >= 0:
                    if abs(x_high - x_low) <= 1e-12:
                        x_tl = float(x_low)
                    else:
                        sol = root_scalar(objective, bracket=[x_low, x_high], method='brentq')
                        x_tl = float(sol.root)
                    _, y_tl_val, Yc = calc_tangent_dist(x_tl)
                    x_tr = 2.0 * a - x_tl

                    n_left = max(8, int(np.ceil(x_tl / effective_step)))
                    x_left = np.linspace(0.0, x_tl, n_left)
                    y_left, _ = _superellipse_eval_local(x_left, a, b, n)

                    phi_l = math.atan2(y_tl_val - Yc, x_tl - a)
                    phi_r = math.atan2(y_tl_val - Yc, x_tr - a)
                    sweep = phi_r - phi_l
                    # Surface fillets retain their exact tangent solution.
                    # Extending this local fillet to 180 degrees and then
                    # pinning it back to x_tr creates a long crossing chord.
                    arc_len = R_min * abs(sweep)
                    n_arc_seg = max(10, int(np.ceil(arc_len / effective_step)))
                    phi_arr_seg = np.linspace(phi_l, phi_l + sweep, n_arc_seg)
                    x_arc_seg = a + R_min * np.cos(phi_arr_seg)
                    y_arc_seg = Yc + R_min * np.sin(phi_arr_seg)

                    n_right = max(8, int(np.ceil((2.0 * a - x_tr) / effective_step)))
                    x_right = np.linspace(x_tr, 2.0 * a, n_right)
                    y_right, _ = _superellipse_eval_local(x_right, a, b, n)

                    curve_x_loc = np.concatenate([x_left[:-1], x_arc_seg[:-1], x_right])
                    curve_y_loc = np.concatenate([y_left[:-1], y_arc_seg[:-1], y_right])

                    # Preserve the fitted fillet and its straight-arm guard.
                    # The later spacing optimizer may move only the remaining
                    # straight portions.
                    arc_start = max(0, len(x_left) - 1)
                    arc_end = min(len(curve_x_loc) - 1,
                                  arc_start + len(x_arc_seg) - 1)
                    # Keep the arc fixed, with the configured total guard
                    # split evenly across its two straight-arm sides.
                    lock_pad = max(1, int(math.ceil(
                        0.5 * arc_lock_guard_d * d / effective_step
                    )))
                    locked_mask = np.zeros(len(curve_x_loc), dtype=bool)
                    locked_mask[max(0, arc_start - lock_pad):
                                min(len(locked_mask), arc_end + lock_pad + 1)] = True

                    curve_3d = M[None, :] + np.outer(curve_x_loc - a, X_dir) + np.outer(curve_y_loc, Y_dir)
                    curve_3d[0] = P1
                    curve_3d[-1] = P3

                    # 【核心修复】：工况 A 为表面贴体轨迹，必须无条件投影贴合回曲面
                    # Case A is a surface fillet: it must always remain
                    # attached to the source mesh, regardless of the B-arc
                    # projection policy.
                    curve_3d, _, _ = trimesh.proximity.closest_point(mesh, curve_3d)
                    curve_3d[0] = P1
                    curve_3d[-1] = P3

                    n_pts = len(curve_3d)
                    phases = np.where(np.arange(n_pts) < n_pts // 2, 0, 1).astype(int)

                    fillet_meta = dict(meta)
                    fillet_meta["locked_mask"] = locked_mask.tolist()
                    raw_filleted.append(TrajectorySegment(
                        kind="SURFACE_SCAN",
                        uv=np.full((len(curve_3d), 2), np.nan),
                        xyz=curve_3d,
                        spray_on=True,
                        note=seg.note.split(" [")[0] + f" [Surface Fillet R={R_min:.1f}mm]",
                        region=seg.region,
                        phases=phases,
                        geometry_meta=fillet_meta
                    ))
                    condition_a_success = True
            except Exception:
                condition_a_success = False

        if condition_a_success:
            case_stats["case_a"] += 1
            continue

        # =====================================================================
        # 工况 B：空间 Rodrigues 圆弧倒圆（悬空过渡）
        # =====================================================================
        n_full = max(32, int(np.ceil((2.0 * a) / effective_step)))
        curve_x_loc = np.linspace(0.0, 2.0 * a, n_full)
        curve_y_loc, _ = _superellipse_eval_local(curve_x_loc, a, b, n)
        raw_xyz = M[None, :] + np.outer(curve_x_loc - a, X_dir) + np.outer(curve_y_loc, Y_dir)
        raw_xyz[0] = P1
        raw_xyz[-1] = P3
        # 初始骨架投影贴合表面
        raw_xyz, _, _ = trimesh.proximity.closest_point(mesh, raw_xyz)
        raw_xyz[0] = P1
        raw_xyz[-1] = P3

        p1_pt = raw_xyz[0]
        p3_pt = raw_xyz[-1]
        chord_vec = p3_pt - p1_pt
        chord_len = float(np.linalg.norm(chord_vec))

        if chord_len < 1e-9:
            case_stats["drop_degenerate"] += 1
            continue

        chord_unit = chord_vec / chord_len
        relative = raw_xyz - p1_pt
        proj_len = relative @ chord_unit
        proj_pts = p1_pt + np.outer(proj_len, chord_unit)
        perp_dists = np.linalg.norm(raw_xyz - proj_pts, axis=1)
        idx_tip = int(np.argmax(perp_dists))

        if idx_tip <= 0 or idx_tip >= len(raw_xyz) - 1:
            idx_tip = len(raw_xyz) // 2

        p2_pt = raw_xyz[idx_tip]
        arm1 = p1_pt - p2_pt
        arm2 = p3_pt - p2_pt
        len1 = float(np.linalg.norm(arm1))
        len2 = float(np.linalg.norm(arm2))

        if len1 < 1e-9 or len2 < 1e-9:
            case_stats["drop_degenerate"] += 1
            continue

        u1 = arm1 / len1
        u2 = arm2 / len2

        cos_theta = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = math.acos(cos_theta)
        theta = np.clip(theta, math.radians(2.0), math.radians(178.0))
        half_theta = theta / 2.0

        sin_half = math.sin(half_theta)
        tan_half = math.tan(half_theta)

        L2 = R_min / max(tan_half, 1e-4)
        L1 = R_min / max(sin_half, 1e-4)

        bisector = u1 + u2
        bisector_len = float(np.linalg.norm(bisector))
        u_bisector = Y_dir if bisector_len < 1e-9 else (bisector / bisector_len)

        center = p2_pt + u_bisector * L1

        # Empirical boundary rule: a center may be outside the mesh and still
        # produce a valid tangent arc, but once its outside distance exceeds
        # the configured limit the arc is geometrically unreliable and is
        # discarded.  This is deliberately a magnitude limit, not a boolean
        # "center outside => invalid" test.
        _, center_dist_arr, _ = trimesh.proximity.closest_point(mesh, center[None, :])
        center_dist = float(center_dist_arr[0])
        if center_dist > center_limit:
            case_stats["drop_center_limit"] += 1
            print(
                f"[彻底删除] 圆心越界={center_dist:.1f}mm > "
                f"设定上限={center_limit:.1f}mm "
                f"(θ={math.degrees(theta):.1f}°)，已删除该圆弧。"
            )
            continue

        T1 = p2_pt + u1 * L2
        T2 = p2_pt + u2 * L2

        v_start = T1 - center
        v_end = T2 - center
        u_start = v_start / max(np.linalg.norm(v_start), 1e-9)
        u_end = v_end / max(np.linalg.norm(v_end), 1e-9)

        cross_v = np.cross(u_start, u_end)
        cross_len = float(np.linalg.norm(cross_v))

        if cross_len < 1e-9:
            rot_axis = np.cross(u_start, u_bisector)
            rot_axis /= max(np.linalg.norm(rot_axis), 1e-9)
        else:
            rot_axis = cross_v / cross_len

        tangent_angle = math.pi - theta
        # P1/P3 in condition B are the new endpoints of one semicircle, not
        # the endpoints of the fitted source track.  Extend the same circle on
        # both sides of the exact tangent interval T1->T2 so that the complete
        # order is P1 -> T1 -> T2 -> P3 and the total sweep is exactly 180 deg.
        extension_angle = 0.5 * (math.pi - tangent_angle)
        n_before = max(2, int(math.ceil(R_min * extension_angle / effective_step)) + 1)
        n_between = max(3, int(math.ceil(R_min * tangent_angle / effective_step)) + 1)
        n_after = n_before
        phi_before = np.linspace(-extension_angle, 0.0, n_before)
        phi_between = np.linspace(0.0, tangent_angle, n_between)
        phi_after = np.linspace(tangent_angle, tangent_angle + extension_angle, n_after)
        phi_values = np.concatenate([phi_before[:-1], phi_between[:-1], phi_after])
        t1_index = n_before - 1
        t2_index = t1_index + n_between - 1

        arc_points = np.empty((len(phi_values), 3), dtype=float)
        for k, phi in enumerate(phi_values):
            rotated = (
                u_start * math.cos(phi)
                + np.cross(rot_axis, u_start) * math.sin(phi)
                + rot_axis * np.dot(rot_axis, u_start) * (1.0 - math.cos(phi))
            )
            arc_points[k] = center + R_min * rotated

        arc_points[t1_index] = T1
        arc_points[t2_index] = T2

        # B 区不能一律按悬空处理：如果 T1->T2 的中点位于原始 a1-a2
        # 弦的模型侧，则这是部分落在曲面上的边界圆弧，只投影真正贴近
        # 曲面的样本；只有整段位于模型外侧时才完全不投影。
        # Case B projection policy.  The source V-track endpoints are p1/p3;
        # use their midpoint as the reference centre.  A partial arc is only
        # considered sufficiently inside the surface when its penetration
        # depth is at least 2*d, i.e. |C-M| <= R-2*d.  Shallower partial arcs
        # are retained entirely as the analytic free-space circle.
        chord_midpoint = 0.5 * (p1_pt + p3_pt)
        center_to_midpoint = float(np.linalg.norm(center - chord_midpoint))
        deep_enough_for_projection = center_to_midpoint <= (R_min - 2.0 * d)
        if deep_enough_for_projection:
            arc_points, b_arc_has_surface_part = project_partial_b_arc(
                arc_points,
                t1_index,
                t2_index,
                p1_pt,
                p3_pt,
                p2_pt,
            )
        else:
            # Fully outside, or only shallowly inside (<2d): preserve the
            # complete analytic circle/fillet without any projection.
            b_arc_has_surface_part = False
        case_stats["case_b"] += 1
        arc_points[t1_index] = T1
        arc_points[t2_index] = T2
        final_xyz = arc_points

        if len(final_xyz) >= 2:
            edges = np.linalg.norm(np.diff(final_xyz, axis=0), axis=1)
            keep_mask = np.r_[True, edges > 1e-7]
            final_xyz = final_xyz[keep_mask]

        # Lock the analytic B-arc plus the configured side guard before global
        # spacing relaxation.  Apply the same duplicate-point mask so the
        # metadata remains aligned with final_xyz.
        # Keep the arc fixed, with the configured total guard split evenly
        # across its two straight-arm sides.
        lock_pad = max(1, int(math.ceil(
            0.5 * arc_lock_guard_d * d / effective_step
        )))
        locked_mask = np.zeros(len(arc_points), dtype=bool)
        lock_lo = max(0, int(t1_index) - lock_pad)
        lock_hi = min(len(locked_mask), int(t2_index) + lock_pad + 1)
        locked_mask[lock_lo:lock_hi] = True
        if len(final_xyz) != len(locked_mask):
            locked_mask = locked_mask[keep_mask]

        total_len = float(np.sum(np.linalg.norm(np.diff(final_xyz, axis=0), axis=1))) if len(final_xyz) >= 2 else 0.0
        if total_len < 0.8 * d:
            print(f"[过短删除] 轨迹长度={total_len:.2f}mm < 0.8d({0.8*d:.2f}mm)，已剔除该残余段。")
            case_stats["drop_short"] += 1
            continue
        mid_idx = len(final_xyz) // 2
        phases = np.where(np.arange(len(final_xyz)) < mid_idx, 0, 1).astype(int)

        fillet_meta = dict(meta)
        fillet_meta["locked_mask"] = locked_mask.tolist()
        raw_filleted.append(TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=np.full((len(final_xyz), 2), np.nan),
            xyz=final_xyz,
            spray_on=True,
            note=seg.note.split(" [")[0] + f" [Spatial Arc R={R_min:.1f}mm; {'surface-partial' if b_arc_has_surface_part else 'air-only'}]",
            region=seg.region,
            phases=phases,
            geometry_meta=fillet_meta
        ))

    # =========================================================================
    # 拓扑重构与 OVERTRAVEL 缝合
    # =========================================================================
    cleaned_segments: List[TrajectorySegment] = []
    for seg in raw_filleted:
        if seg.kind == "SURFACE_SCAN":
            if len(cleaned_segments) > 0 and cleaned_segments[-1].kind == "SURFACE_SCAN":
                overtravel_seg = _overtravel_segment(
                    cleaned_segments[-1], seg, overtravel_factor * target_spacing
                )
                cleaned_segments.append(overtravel_seg)
            elif len(cleaned_segments) > 0 and cleaned_segments[-1].kind == "OVERTRAVEL":
                prev_ot = cleaned_segments[-1]
                prev_ot.xyz[-1] = seg.xyz[0]
                if len(prev_ot.xyz) >= 4:
                    prev_ot.xyz[2] = seg.xyz[0] + (prev_ot.xyz[1] - prev_ot.xyz[0])
            cleaned_segments.append(seg)
        elif seg.kind == "OVERTRAVEL":
            if len(cleaned_segments) > 0 and cleaned_segments[-1].kind == "SURFACE_SCAN":
                cleaned_segments.append(seg)

    while len(cleaned_segments) > 0 and cleaned_segments[-1].kind == "OVERTRAVEL":
        cleaned_segments.pop()

    # Filleting can move a surface endpoint after its original connector was
    # created.  Rebuild every connector from the final neighbouring surfaces
    # so both ends are coincident (zero positional gap) in the rendered and
    # exported trajectory.
    repaired_segments: List[TrajectorySegment] = []
    for index, seg in enumerate(cleaned_segments):
        if (
                seg.kind == "OVERTRAVEL"
                and index > 0
                and index + 1 < len(cleaned_segments)
                and cleaned_segments[index - 1].kind == "SURFACE_SCAN"
                and cleaned_segments[index + 1].kind == "SURFACE_SCAN"
        ):
            repaired_segments.append(_overtravel_segment(
                cleaned_segments[index - 1], cleaned_segments[index + 1],
                overtravel_factor * target_spacing,
            ))
        else:
            repaired_segments.append(seg)

    print(
        "[Fillet case summary] "
        f"input={case_stats['input_superellipse']}, "
        f"A={case_stats['case_a']}, B={case_stats['case_b']}, "
        f"drop_center={case_stats['drop_center_limit']}, "
        f"drop_short={case_stats['drop_short']}, "
        f"drop_degenerate={case_stats['drop_degenerate']}, "
        f"passthrough={case_stats['passthrough']}"
    )
    return repaired_segments
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


def _trajectory_objective(
        tri_centers: np.ndarray,
        fixed_pts: np.ndarray,
        lines: Sequence[np.ndarray],
        target_spacing: float,
        coverage_weight: float,
        spacing_weight: float,
        boundary_weight: float,
        smooth_weight: float,
        radial_barriers: Sequence[dict[str, object]] | None = None,
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

    # Use the same non-uniform radial barrier as the actual CENTER update.
    if radial_barriers:
        barrier_force, _, _, _, _ = _radial_barrier_state(
            center_pts, radial_barriers, d, stiffness=1.0
        )
        boundary_loss = float(np.mean(np.linalg.norm(barrier_force, axis=1) ** 2))
    elif len(fixed_pts):
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


def _fixed_spline_samples(
        trajectory: Sequence[TrajectorySegment],
        fixed_indices: Sequence[int],
        sample_step: float,
) -> np.ndarray:
    """Sample the fitted LOWER/UPPER splines for geometry-independent energies."""
    samples = []
    for index in fixed_indices:
        xyz = np.asarray(trajectory[index].xyz, dtype=float)
        if len(xyz) >= 2:
            samples.append(_resample_polyline(xyz, sample_step))
        elif len(xyz) == 1:
            samples.append(xyz)
    return np.vstack(samples) if samples else np.empty((0, 3), dtype=float)


def _smooth_linewise_vectors(
    vectors: np.ndarray,
    offsets: np.ndarray,
    passes: int = 2,
    radius: int = 2,
) -> np.ndarray:
    """Low-pass a vector field independently along each trajectory."""
    result = np.asarray(vectors, dtype=float).copy()
    radius = max(1, int(radius))
    samples = np.arange(-radius, radius + 1, dtype=float)
    sigma = max(1.0, 0.5 * radius)
    kernel = np.exp(-0.5 * (samples / sigma) ** 2)
    kernel /= float(np.sum(kernel))
    for _ in range(max(0, int(passes))):
        previous = result.copy()
        for line_index in range(len(offsets) - 1):
            start, stop = int(offsets[line_index]), int(offsets[line_index + 1])
            if stop - start < len(kernel):
                continue
            padded = np.pad(
                previous[start:stop], ((radius, radius), (0, 0)), mode="edge"
            )
            for axis in range(3):
                result[start:stop, axis] = np.convolve(
                    padded[:, axis], kernel, mode="valid"
                )
    return result


def _directed_pair_force(
    follower: np.ndarray,
    leader: np.ndarray,
    spacing: float,
    attract_weight: float,
    repel_weight: float,
    gain: float = 1.0,
) -> np.ndarray:
    """Return a d-equilibrium force applied only to the follower line."""
    follower = np.asarray(follower, dtype=float)
    leader = np.asarray(leader, dtype=float)
    force = np.zeros_like(follower)
    if not len(follower) or not len(leader):
        return force
    distance, nearest = cKDTree(leader).query(follower)
    active = distance > 1.0e-6
    if not np.any(active):
        return force
    d = max(float(spacing), EPS)
    gap = distance[active]
    delta = follower[active] - leader[nearest[active]]
    direction = delta / gap[:, None]
    error = (gap - d) / d
    dead = np.clip(np.abs(error) / 0.04, 0.0, 1.0)
    dead = dead * dead * (3.0 - 2.0 * dead)
    magnitude = np.where(
        gap < d,
        repel_weight * np.minimum(d / gap - 1.0, 4.0),
        -attract_weight * np.minimum(gap / d - 1.0, 1.0),
    ) * dead * float(gain)
    force[active] = direction * magnitude[:, None]
    return force


def _blank_guidance_force(
    points: np.ndarray,
    centers: Sequence[np.ndarray],
    spacing: float,
    influence_radius: float,
    gain: float,
) -> np.ndarray:
    """Long-range guidance that fades smoothly as points reach a blank center."""
    points = np.asarray(points, dtype=float)
    force = np.zeros_like(points)
    d = max(float(spacing), EPS)
    response_radius = max(float(influence_radius), 0.75 * d, EPS)
    dead_radius = 0.5 * d
    for center in centers:
        delta = np.asarray(center, dtype=float) - points
        distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-8)
        response_t = np.clip(
            (distance - dead_radius)
            / max(response_radius - dead_radius, EPS),
            0.0,
            1.0,
        )
        response = response_t ** 2 * (3.0 - 2.0 * response_t)
        force += (
            float(gain) * response
        )[:, None] * (delta / distance[:, None])
    return force


def _remove_leader_gap_components(
    force: np.ndarray,
    points: np.ndarray,
    offsets: np.ndarray,
) -> None:
    """Prevent guidance force from changing the C0/C_last separation."""
    if len(offsets) < 3:
        return
    first_start, first_stop = int(offsets[0]), int(offsets[1])
    last_start, last_stop = int(offsets[-2]), int(offsets[-1])
    first = points[first_start:first_stop]
    last = points[last_start:last_stop]
    if not len(first) or not len(last):
        return

    _, nearest_last = cKDTree(last).query(first)
    first_normal = first - last[nearest_last]
    first_norm = np.linalg.norm(first_normal, axis=1)
    valid_first = first_norm > 1.0e-8
    first_normal[valid_first] /= first_norm[valid_first, None]
    first_force = force[first_start:first_stop]
    first_component = np.einsum("ij,ij->i", first_force, first_normal)
    first_force[valid_first] -= (
        first_component[valid_first, None] * first_normal[valid_first]
    )

    _, nearest_first = cKDTree(first).query(last)
    last_normal = last - first[nearest_first]
    last_norm = np.linalg.norm(last_normal, axis=1)
    valid_last = last_norm > 1.0e-8
    last_normal[valid_last] /= last_norm[valid_last, None]
    last_force = force[last_start:last_stop]
    last_component = np.einsum("ij,ij->i", last_force, last_normal)
    last_force[valid_last] -= (
        last_component[valid_last, None] * last_normal[valid_last]
    )


def _prioritize_leader_gap_displacement(
    displacement: np.ndarray,
    base_displacement: np.ndarray,
    points: np.ndarray,
    offsets: np.ndarray,
    step_cap: np.ndarray,
) -> None:
    """Reserve leader normal motion for non-blank forces; fit guidance tangentially."""
    if len(offsets) < 3:
        return
    first_start, first_stop = int(offsets[0]), int(offsets[1])
    last_start, last_stop = int(offsets[-2]), int(offsets[-1])
    first = points[first_start:first_stop]
    last = points[last_start:last_stop]

    def enforce(start: int, stop: int, normal: np.ndarray) -> None:
        normal_norm = np.linalg.norm(normal, axis=1)
        valid = normal_norm > 1.0e-8
        normal[valid] /= normal_norm[valid, None]
        target = displacement[start:stop]
        base = base_displacement[start:stop]
        cap = np.asarray(step_cap[start:stop], dtype=float)
        target_normal = np.einsum("ij,ij->i", target, normal)
        desired_normal = np.clip(
            np.einsum("ij,ij->i", base, normal), -cap, cap
        )
        tangent = target - target_normal[:, None] * normal
        tangent_norm = np.linalg.norm(tangent, axis=1)
        tangent_limit = np.sqrt(np.maximum(cap ** 2 - desired_normal ** 2, 0.0))
        tangent_scale = np.minimum(
            1.0, tangent_limit / np.maximum(tangent_norm, 1.0e-12)
        )
        target[valid] = (
            tangent[valid] * tangent_scale[valid, None]
            + desired_normal[valid, None] * normal[valid]
        )

    _, nearest_last = cKDTree(last).query(first)
    enforce(first_start, first_stop, first - last[nearest_last])
    _, nearest_first = cKDTree(first).query(last)
    enforce(last_start, last_stop, last - first[nearest_first])


def _optimize_trajectory_region_energy(
        trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        moving_region: str,
        iterations: int = 140,
        attract_weight: float = 0.35,  # 空白区引导引力
        self_repel_weight: float = 0.30,  # 线上节点互斥力
        arc_repel_weight: float = 3.50,  # 固定样条边界排斥权重（保留参数名以兼容旧调用）
        elastic_weight: float = 0.35,  # 顺向平滑刚度
        transverse_weight: float = 0.60,  # 跨线 Δ 变动均摊刚度
        pair_attract_weight: float = 6.00,
        pair_repel_weight: float = 6.00,
        target_spacing: float = 30.0,
        sample_step: float | None = None,
        influence_radius: float | None = None,
        blank_centers_3d: dict[str, Sequence[float]] | None = None,
        periodic_pair_blocked: bool | None = None,
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
    """Optimize one scan region without coupling it to other movable regions."""
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

    moving_region = str(moving_region).upper()
    if moving_region not in {"CENTER", "LOWER", "UPPER"}:
        raise ValueError("moving_region must be 'CENTER', 'LOWER', or 'UPPER'.")

    lower_idx = [i for i, s in enumerate(trajectory)
                 if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "LOWER"]
    upper_idx = [i for i, s in enumerate(trajectory)
                 if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "UPPER"]
    center_idx = [i for i, s in enumerate(trajectory)
                  if s.kind == "SURFACE_SCAN" and getattr(s, "region", "") == "CENTER"]
    if moving_region == "CENTER":
        moving_idx = center_idx
    else:
        red_candidates = lower_idx if moving_region == "LOWER" else upper_idx
        # A red region needs at least three paths before spatial ordering is
        # meaningful.  Select the four paths closest to the CENTER-side
        # boundary, using whole-path mean distance rather than a local spike.
        # Store them far-to-near so Line 0 is the outer reference path.
        if len(red_candidates) >= 3 and center_idx:
            boundary_center_idx = center_idx
            if len(center_idx) > 1:
                red_blocks = [
                    np.asarray(trajectory[index].xyz, dtype=float)
                    for index in red_candidates
                    if len(trajectory[index].xyz) > 0
                ]
                all_red_pts = np.vstack(red_blocks) if red_blocks else np.empty((0, 3), dtype=float)
                red_tree = cKDTree(all_red_pts) if len(all_red_pts) else None
                if red_tree is not None:
                    valid_centers = [
                        index for index in center_idx
                        if len(trajectory[index].xyz) > 0
                    ]
                    if valid_centers:
                        closest_center_idx = min(
                            valid_centers,
                            key=lambda index: float(np.mean(red_tree.query(
                                np.asarray(trajectory[index].xyz, dtype=float)
                            )[0])),
                        )
                        boundary_center_idx = [closest_center_idx]

            boundary_samples = _fixed_spline_samples(
                trajectory, boundary_center_idx, step0
            )
            boundary_tree = cKDTree(boundary_samples) if len(boundary_samples) else None#选择最靠近红色区域的 CENTER 轨迹  
            ranked = []
            for index in red_candidates:
                points = np.asarray(trajectory[index].xyz, dtype=float)
                if boundary_tree is None or len(points) == 0:
                    distance = float("inf")
                else:
                    distance = float(np.mean(boundary_tree.query(points)[0]))
                ranked.append((distance, index))
            ranked.sort(key=lambda item: item[0])
            moving_idx = [index for _, index in ranked[:4]][::-1]
        else:
            # With zero, one, or two candidates keep their established order;
            # there is no reliable multi-line ordering to infer.
            moving_idx = red_candidates[:4]

    if not moving_idx:
        print(f"[Energy optimizer skipped] no SURFACE_SCAN segment is tagged {moving_region}.")
        return trajectory

    # Only one region moves in a pass. The other scan region remains a fixed
    # boundary/reference, so the two passes are independent in their forces
    # while still preserving the required inter-region clearance.
    if moving_region == "CENTER":
        fixed_idx = lower_idx + upper_idx
    else:
        # Both red boundary groups move toward CENTER. Do not include the
        # opposite red group here, otherwise the first line reaching d hides
        # the remaining gaps from the actual CENTER boundary.
        all_red = lower_idx + upper_idx
        fixed_idx = center_idx + [index for index in all_red if index not in moving_idx]

    # LOWER/UPPER 已是“直线 + 圆弧”过渡的五次 B 样条。用其采样点定义
    # 边界势能，避免再从端点反推一个并不存在的半圆。
    lines = [_resample_polyline(np.asarray(trajectory[i].xyz, float), step0)
             for i in moving_idx]
    locked_masks = []
    for segment_index, line in zip(moving_idx, lines):
        if moving_region == "CENTER":
            locked_masks.append(np.zeros(len(line), dtype=bool))
            continue
        stored = np.asarray(
            (trajectory[segment_index].geometry_meta or {}).get("locked_mask", []),
            dtype=bool,
        )
        if len(stored) == len(line):
            locked_masks.append(stored.copy())
        elif len(stored) > 1:
            line_u = np.linspace(0.0, 1.0, len(line))
            stored_u = np.linspace(0.0, 1.0, len(stored))
            locked_masks.append(np.interp(
                line_u, stored_u, stored.astype(float)
            ) >= 0.5)
        else:
            locked_masks.append(np.zeros(len(line), dtype=bool))
    fixed_pts = _fixed_spline_samples(trajectory, fixed_idx, step0)
    fixed_tree = cKDTree(fixed_pts) if len(fixed_pts) else None
    radial_barriers: list[dict[str, object]] = []
    if moving_region == "CENTER":
        hard_boundary_indices = []
        if lower_idx:
            hard_boundary_indices.append(("LOWER", lower_idx[-1]))
        if upper_idx:
            hard_boundary_indices.append(("UPPER", upper_idx[0]))
        for side, boundary_index in hard_boundary_indices:
            raw_boundary_points = np.asarray(
                trajectory[boundary_index].xyz, float
            )
            boundary_points = _resample_polyline(
                raw_boundary_points, step0
            )
            if len(boundary_points) < 2:
                continue
            center = _project_to_mesh(
                mesh, _closed_polyline_center(raw_boundary_points)
            )
            if center is None:
                continue
            radial_barriers.append({
                "side": side,
                "center": center,
                "points": boundary_points,
                "tree": cKDTree(boundary_points),
            })
    print(
        f"[Energy optimizer] region={moving_region}, movable_paths={len(moving_idx)}, "
        f"arc_anchors={len(fixed_pts)}, iterations={iterations}, d={d:.6g}",
        flush=True,
    )
    tri_centers = np.asarray(mesh.triangles_center, dtype=float)
    initial_line_trees = [cKDTree(line) for line in lines]
    endpoints = [(L[0].copy(), L[-1].copy()) for L in lines]
    num_lines = len(lines)
    middle_anchor_index = (
        num_lines // 2
        if moving_region == "CENTER" and num_lines >= 3 and num_lines % 2 == 1
        else None
    )

    # =========================================================
    # 新增：用于均匀性判断的固定喷涂点、顶点树、顶点面积权重
    # =========================================================
    moving_spray_flags = [
        bool(getattr(trajectory[si], "spray_on", True))
        for si in moving_idx
    ]

    fixed_spray_pts = fixed_pts if uniform_stop else np.empty((0, 3), dtype=float)

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

    blanks: list[np.ndarray] = []
    void_depth = 0.0
    if moving_region == "CENTER":
        center_geometry = None
        if blank_centers_3d is None or periodic_pair_blocked is None:
            center_geometry = _special_boundary_geometry(
                trajectory, d, mesh=mesh
            )
        fixed_blank_centers = blank_centers_3d
        if fixed_blank_centers is None:
            fixed_blank_centers = center_geometry.get(
                "blank_centers_3d", {}
            )
        if periodic_pair_blocked is None:
            periodic_pair_blocked = bool(
                center_geometry.get("periodic_pair_blocked", False)
            )
        if isinstance(fixed_blank_centers, dict):
            for side in ("LOWER", "UPPER"):
                value = fixed_blank_centers.get(side)
                if value is None:
                    continue
                point = np.asarray(value, dtype=float)
                if point.shape == (3,) and np.isfinite(point).all():
                    blanks.append(point)
        if blanks:
            initial_coverage = np.vstack(
                [fixed_pts, np.vstack(lines)] if len(fixed_pts) else lines
            )
            if len(initial_coverage):
                void_depth = float(np.max(
                    cKDTree(initial_coverage).query(np.asarray(blanks))[0]
                ))
    periodic_pair_blocked = bool(periodic_pair_blocked)

    for it in range(iterations):
        P = np.vstack(lines)
        line_id = np.concatenate([np.full(len(L), li, np.int64) for li, L in enumerate(lines)])
        local_id = np.concatenate([np.arange(len(L)) for L in lines])
        offsets = np.r_[0, np.cumsum([len(L) for L in lines])]
        # Arc samples and their d-wide side guards are immutable.  Excluding
        # them from all forces also prevents a neighbouring spring from
        # distorting the straight-to-arc transition before it is restored.
        moving_point = np.ones(len(P), dtype=bool)
        if moving_region != "CENTER":
            moving_point = ~np.concatenate(locked_masks)#也是固定参考点，会被其他center点排斥力影响
        elif middle_anchor_index is not None:
            anchor_start = int(offsets[middle_anchor_index])
            anchor_stop = int(offsets[middle_anchor_index + 1])
            moving_point[anchor_start:anchor_stop] = False

        large_void = void_depth > 1.2 * d
        R = float(influence_radius) if influence_radius is not None \
            else (4.5 * d if large_void else 3.0 * d)

        # Keep coverage motion separate from the slower coupled-spacing
        # relaxation.  Only CENTER receives the former.
        F = np.zeros_like(P)
        void_force = np.zeros_like(P)

        # --- A. 空白区引导引力 ---
        if moving_region == "CENTER":
            # Each CENTER point owns its distance response.  Points inside
            # 0.5d have no void force; a nearby point must not disable the
            # long-range guidance acting on farther points.
            depth_excess = np.clip(
                (void_depth - 0.75 * d) / max(d, EPS), 0.0, 3.0
            )
            void_gain = (
                attract_weight * (1.0 + depth_excess ** 1.2)
            )
            void_force = _blank_guidance_force(
                P, blanks, d, R, void_gain
            )

        # The selected red paths form a spring chain ordered far-to-near from
        # CENTER.  Line 0 is the outer reference; each progressively closer
        # path relaxes toward d from its outer neighbour without applying a
        # spring force back into that reference chain.//this prat for lower and upper region
        neighbour_links = set()
        if moving_region != "CENTER" and num_lines >= 2:
            line_trees = [cKDTree(line) for line in lines]
            pairs = {(li, li + 1) for li in range(num_lines - 1)}
            neighbour_links = pairs
            for li, lj in pairs:
                a, b = int(offsets[lj]), int(offsets[lj + 1])
                _, closest = line_trees[li].query(P[a:b])
                delta = P[a:b] - lines[li][closest]
                distance = np.linalg.norm(delta, axis=1)
                deadband = 0.04 * d
                active_pair = (distance > 1.0e-6) & moving_point[a:b]
                if not np.any(active_pair):
                    continue
                direction = delta[active_pair] / distance[active_pair, None]
                # The pair force is zero at d. Compression receives a much
                # stronger barrier than the long-range attraction.
                gap = distance[active_pair]
                error = (gap - d) / max(d, EPS)
                smooth = np.clip(np.abs(error) / max(deadband / d, EPS), 0.0, 1.0)
                smooth = smooth * smooth * (3.0 - 2.0 * smooth)
                magnitude = np.where(
                    gap < d,
                    pair_repel_weight * np.minimum(d / gap - 1.0, 4.0),
                    -pair_attract_weight * np.minimum(gap / d - 1.0, 1.0),
                ) * smooth
                local = np.flatnonzero(active_pair)
                F[a + local] += direction * magnitude[:, None]

        tc = cKDTree(P)

        # --- B. Legacy fixed-reference force for red-path relaxation. ---
        # CENTER bypasses this soft nearest-point field and receives the
        # radial hard barrier after all ordinary forces are assembled.
        if fixed_tree is not None and moving_region != "CENTER":
            boundary_clearance = 1.15 * d
            boundary_distance, nearest = fixed_tree.query(
                P, distance_upper_bound=boundary_clearance
            )
            active = (
                moving_point
                & np.isfinite(boundary_distance)
                & (nearest < len(fixed_pts))
            )
            if np.any(active):
                away = P[active] - fixed_pts[nearest[active]]
                away_norm = np.linalg.norm(away, axis=1)
                nonzero = away_norm > 1.0e-6
                if np.any(nonzero):
                    active_indices = np.flatnonzero(active)[nonzero]
                    magnitude = arc_repel_weight * (
                        (boundary_clearance - boundary_distance[active][nonzero]) / d
                    )
                    F[active_indices] += (
                        away[nonzero] / away_norm[nonzero, None] * magnitude[:, None]
                    )

        # --- C. 同线与局部节点自排斥力 ---
        coo = tc.sparse_distance_matrix(tc, 1.8 * d, output_type="coo_matrix")
        r_, c_, dv = coo.row, coo.col, coo.data
        ok = (dv > 1e-6) & (r_ != c_)
        same = line_id[r_] == line_id[c_]
        ok &= ~(same & (np.abs(local_id[r_] - local_id[c_]) <= win))
        # Cross-line interaction belongs exclusively to the directed Fpair
        # chain below. Self-repulsion is local to one trajectory, so an inner
        # follower can never push force back into C0 or C_last.
        ok &= same
        r_, c_, dv = r_[ok], c_[ok], dv[ok]
        mag = self_repel_weight * (1.0 - dv / (1.8 * d))
        np.add.at(F, c_, (P[c_] - P[r_]) / dv[:, None] * mag[:, None])

        # --- D. 沿线纵向顺向平滑 ---
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            if b - a > 2:
                F[a + 1:b - 1] += elastic_weight * (P[a:b - 2] - 2 * P[a + 1:b - 1] + P[a + 2:b])

        # --- E. Directed asymmetric CENTER coupling (Fpair). ---
        if moving_region == "CENTER" and num_lines >= 2:
            def apply_follower(follower_index: int, leader_index: int, gain: float = 1.0) -> None:
                start, stop = int(offsets[follower_index]), int(offsets[follower_index + 1])
                F[start:stop] += _directed_pair_force(
                    P[start:stop],
                    lines[leader_index],
                    d,
                    pair_attract_weight,
                    pair_repel_weight,
                    gain=transverse_weight * gain,
                )

            # C0 and C_last remain periodic neighbours unless an independent
            # physical-barrier test explicitly blocks their interaction.
            if not periodic_pair_blocked:
                apply_follower(0, num_lines - 1)
                apply_follower(num_lines - 1, 0)

            anchor_indices = {0, num_lines - 1}
            if middle_anchor_index is not None:
                anchor_indices.add(middle_anchor_index)

            # Every non-anchor line balances its own two gaps. A large gap
            # attracts it toward that side; a compressed gap repels it. The
            # force is never written into either neighbour, so the three
            # anchors cannot receive feedback from followers.
            for follower in range(1, num_lines - 1):
                if follower in anchor_indices:
                    continue
                apply_follower(follower, follower - 1, gain=0.5)
                apply_follower(follower, follower + 1, gain=0.5)

        # Smooth both ordinary forces and green-center attraction. Previously
        # the latter bypassed smoothing and created pointwise kinks.
        force_smoothing_radius = max(2, int(np.ceil(d / step0)))
        displacement_smoothing_radius = max(
            force_smoothing_radius, int(np.ceil(1.5 * d / step0))
        )
        F = _smooth_linewise_vectors(
            F, offsets, passes=2, radius=force_smoothing_radius
        )
        if moving_region == "CENTER":
            void_force = _smooth_linewise_vectors(
                void_force,
                offsets,
                passes=2,
                radius=displacement_smoothing_radius,
            )
            _remove_leader_gap_components(
                void_force, P, offsets
            )

        # --- F. High-priority radial hard barrier for CENTER. ---red->out 
        barrier_force = np.zeros_like(F)
        barrier_direction = np.zeros_like(F)
        barrier_danger = np.zeros(len(P), dtype=bool)
        barrier_inside = np.zeros(len(P), dtype=bool)
        escape_step = np.zeros(len(P), dtype=float)
        if moving_region == "CENTER" and radial_barriers:
            (
                barrier_force,
                barrier_direction,
                barrier_danger,
                barrier_inside,
                escape_step,
            ) = _radial_barrier_state(
                P, radial_barriers, d, arc_repel_weight
            )
            barrier_danger &= moving_point
            barrier_inside &= moving_point
            escape_step[~moving_point] = 0.0

            inward_component = np.einsum("ij,ij->i", F, barrier_direction)
            remove_inward = barrier_danger & (inward_component < 0.0)
            F[remove_inward] -= (
                inward_component[remove_inward, None]
                * barrier_direction[remove_inward]
            )
            void_force[barrier_danger] = 0.0
            F += barrier_force

        # --- G. Displacement caps and annealing. ---F end
        # A small relaxation step prevents a compressed pair from jumping
        # across its d-equilibrium in one iteration.
        # Dampen LOWER/UPPER motion as each pair approaches its d equilibrium.
        pair_error = np.zeros(len(P), dtype=float)
        if num_lines >= 2:
            for li, lj in neighbour_links:
                # Match the one-way spring: only the CENTER-side path lj
                # receives a spacing-driven step; its outer neighbour li is
                # the reference for this link.
                a, b = int(offsets[lj]), int(offsets[lj + 1])
                _, near = cKDTree(lines[li]).query(P[a:b])
                pair_error[a:b] = np.maximum(
                    pair_error[a:b], np.abs(np.linalg.norm(P[a:b] - lines[li][near], axis=1) - d)
                )
        # Keep a meaningful response for moderate spacing errors.  The old
        # 8% floor made red trajectories appear nearly stationary once their
        # gap was within a fraction of d.
        damping = np.clip(pair_error / max(d, 1.0), 0.16, 1.0)
        # LOWER/UPPER and the coupled spacing component use a small relaxation
        # step.  The CENTER blank-coverage component is applied separately
        # below, so a spacing force cannot cancel its fast coverage motion.
        # Red boundary tracks need a visible but controlled response. Their
        # step is larger than the old 1%-8% range while still below CENTER's
        # fast coverage step to avoid overshoot and bending.
        slow_step = d * (0.07 + 0.33 * (anneal ** it)) * damping
        fast_step = d * (0.05 + 0.35 * (anneal ** it))
        # CENTER follows the fast Aeigenopt step. Red boundary paths use the
        # damped relaxation step to avoid overshoot and severe bending.
        if moving_region != "CENTER":
            # Arc samples and their configured side guards remain fixed;
            # only the remaining straight-arm samples relax.
            for li in range(num_lines):
                a, b = int(offsets[li]), int(offsets[li + 1])
                response = np.where(locked_masks[li], 0.0, 0.75)
                F[a:b] *= response[:, None]
        F[~moving_point] = 0.0
        void_force[~moving_point] = 0.0
        center_force = F + void_force if moving_region == "CENTER" else F
        # Force-based progress metric: sum of the Euclidean magnitudes of
        # the forces actually applied to all nodes in this regional pass.
        # This is intentionally distinct from the geometric objective below.
        force_norms = np.linalg.norm(center_force, axis=1)
        force_total = float(np.sum(force_norms))
        force_mean = float(np.mean(force_norms)) if len(force_norms) else 0.0
        force_max = float(np.max(force_norms)) if len(force_norms) else 0.0
        center_norm = np.maximum(np.linalg.norm(center_force, axis=1), 1e-12)
        center_step = (np.full(len(P), fast_step)
                       if moving_region == "CENTER" else slow_step)
        if moving_region == "CENTER" and np.any(barrier_danger):
            center_step[barrier_danger] = escape_step[barrier_danger]
        base_norm = np.maximum(np.linalg.norm(F, axis=1), 1.0e-12)
        base_displacement = F * np.clip(
            center_step / base_norm, 0.0, 1.0
        )[:, None]
        raw_displacement = center_force * np.clip(
            center_step / center_norm, 0.0, 1.0
        )[:, None]
        base_displacement[~moving_point] = 0.0
        raw_displacement[~moving_point] = 0.0
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            base_displacement[a] = 0.0
            base_displacement[b - 1] = 0.0
            raw_displacement[a] = 0.0
            raw_displacement[b - 1] = 0.0

        smoothed_base_displacement = _smooth_linewise_vectors(
            base_displacement,
            offsets,
            passes=2 if moving_region == "CENTER" else 1,
            radius=(
                displacement_smoothing_radius
                if moving_region == "CENTER"
                else force_smoothing_radius
            ),
        )
        displacement = _smooth_linewise_vectors(
            raw_displacement,
            offsets,
            passes=2 if moving_region == "CENTER" else 1,
            radius=(
                displacement_smoothing_radius
                if moving_region == "CENTER"
                else force_smoothing_radius
            ),
        )
        displacement[~moving_point] = 0.0
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            displacement[a] = 0.0
            displacement[b - 1] = 0.0

        # Smoothing must not erase the hard barrier's escape component.
        if moving_region == "CENTER" and np.any(barrier_danger):
            raw_outward = np.einsum(
                "ij,ij->i", raw_displacement, barrier_direction
            )
            smooth_outward = np.einsum(
                "ij,ij->i", displacement, barrier_direction
            )
            required_outward = 0.75 * np.maximum(raw_outward, 0.0)
            restore = barrier_danger & (smooth_outward < required_outward)
            displacement[restore] += (
                (required_outward[restore] - smooth_outward[restore])[:, None]
                * barrier_direction[restore]
            )

        displacement_norm = np.maximum(
            np.linalg.norm(displacement, axis=1), 1.0e-12
        )
        displacement *= np.clip(
            center_step / displacement_norm, 0.0, 1.0
        )[:, None]
        if moving_region == "CENTER" and num_lines >= 2:
            _prioritize_leader_gap_displacement(
                displacement,
                smoothed_base_displacement,
                P,
                offsets,
                center_step,
            )
        P += displacement

        # Remove accumulated geometric kinks before the single mesh
        # projection. Hard-barrier nodes and immutable arc samples remain
        # untouched; their neighbours fair toward them smoothly.
        fairing_locked = (~moving_point) | barrier_danger
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            fairing_locked[a] = True
            fairing_locked[b - 1] = True
        for _ in range(2 if moving_region == "CENTER" else 1):
            previous = P.copy()
            for li in range(num_lines):
                a, b = int(offsets[li]), int(offsets[li + 1])
                if b - a > 2:
                    laplacian = (
                        0.5 * (previous[a:b - 2] + previous[a + 2:b])
                        - previous[a + 1:b - 1]
                    )
                    P[a + 1:b - 1] = (
                        previous[a + 1:b - 1] + 0.45 * laplacian
                    )
            P[fairing_locked] = previous[fairing_locked]

        # 投影回 3D 曲面 + 钉死端点
        P, _, _ = trimesh.proximity.closest_point(mesh, P)
        new_lines = []
        for li in range(num_lines):
            a, b = int(offsets[li]), int(offsets[li + 1])
            L = P[a:b]
            L[0], L[-1] = endpoints[li]
            # Arc/fillet geometry is immutable during spacing relaxation.
            L[locked_masks[li]] = lines[li][locked_masks[li]]
            if middle_anchor_index is not None and li == middle_anchor_index:
                L = lines[li].copy()
            new_lines.append(L)
        lines = new_lines

        # Keep sample indices stable so the stored arc lock masks remain exact.

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
                if moving_spray_flags[li]:
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
        # The live VTK path consumes these snapshots asynchronously.  Emit
        # the first completed step immediately rather than waiting for the
        # former three-iteration batch, which is noticeably slow on large
        # meshes.
        # Send every completed iteration to the VTK main thread.  The worker
        # never touches VTK directly; the timer consumes these snapshots and
        # keeps the visible trajectory synchronized with the optimization.
        if on_iteration is not None:
            # Keep the geometric terms as diagnostics, but expose the force
            # accumulation as the reported iteration energy E.
            _, energy_components = _trajectory_objective(
                tri_centers=tri_centers,
                fixed_pts=fixed_pts,
                lines=lines,
                target_spacing=d,
                coverage_weight=attract_weight,
                spacing_weight=self_repel_weight + transverse_weight,
                boundary_weight=arc_repel_weight,
                smooth_weight=elastic_weight,
                radial_barriers=radial_barriers,
            )
            energy_components = dict(energy_components)
            energy_components["force_total"] = force_total
            energy_components["force_mean"] = force_mean
            energy_components["force_max"] = force_max
            energy_components["boundary_danger_count"] = int(np.count_nonzero(barrier_danger))
            energy_components["boundary_inside_count"] = int(np.count_nonzero(barrier_inside))
            energy_components["boundary_escape_step_max"] = float(np.max(escape_step)) if len(escape_step) else 0.0
            displacement = np.concatenate([
                initial_line_trees[k].query(line)[0]
                for k, line in enumerate(lines)
            ])
            energy_components["center_displacement_mean"] = float(np.mean(displacement))
            energy_components["center_displacement_max"] = float(np.max(displacement))

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

            for k, si in enumerate(moving_idx):
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
                    geometry_meta=orig.geometry_meta,
                    phases=k_phases,  # 【补充】
                )

            preview = _rebuild_overtravel_connectors(preview, 2.0 * d)
            on_iteration(preview, it + 1, force_total, energy_components)

        if should_stop:
            print(f"\n[Uniformity early stop] {stop_reason}")
            break

    # 如果开启了 return_best_uniform，则返回历史最优版本
    if uniform_stop and return_best_uniform and best_lines is not None:
        lines = best_lines

    out = list(trajectory)

    note_suffix = " [Global Spline-Boundary Energy Optimized]"
    if stop_reason is not None:
        note_suffix = " [Global Spline-Boundary Energy Optimized, Uniformity Early Stop]"

    for k, si in enumerate(moving_idx):
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
            geometry_meta=orig.geometry_meta,
            phases=k_phases,  # 【补充】f
        )

    return _rebuild_overtravel_connectors(out, 2.0 * d)


def optimize_center_trajectories_dual_blank_energy(
        trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        **kwargs,
) -> List[TrajectorySegment]:
    """Synchronously optimize CENTER and UPPER in alternating one-step passes.

    Each pass builds its own coverage and force field; the non-moving regions
    are fixed boundaries for that step rather than simultaneously moving force
    sources.
    """
    from scipy.spatial import cKDTree

    on_iteration = kwargs.pop("on_iteration", None)
    iterations = int(kwargs.get("iterations", 140))
    target_spacing = float(kwargs.get("target_spacing", 30.0))
    current_blank_centers = kwargs.pop("blank_centers_3d", None)

    # The stop test applies to the red tracks that are actually allowed to
    # move, not to the permanently fixed outer tracks.
    center_indices = [i for i, s in enumerate(trajectory)
                      if s.kind == "SURFACE_SCAN"
                      and str(getattr(s, "region", "")).upper() == "CENTER"]
    center_samples = _fixed_spline_samples(trajectory, center_indices,
                                           target_spacing / 4.0)
    center_tree = cKDTree(center_samples) if len(center_samples) else None
    moving_red_indices = set()
    for region in ("LOWER", "UPPER"):
        candidates = [i for i, s in enumerate(trajectory)
                      if s.kind == "SURFACE_SCAN"
                      and str(getattr(s, "region", "")).upper() == region]
        ranked = []
        for index in candidates:
            points = np.asarray(trajectory[index].xyz, dtype=float)
            gap = float(np.min(center_tree.query(points)[0])) if center_tree is not None and len(points) else float("inf")
            ranked.append((gap, index))
        moving_red_indices.update(index for _, index in sorted(ranked)[:3])

    # Advance both regions one iteration at a time. This keeps the rendered
    # CENTER and UPPER paths synchronized while each pass still has its own
    # force field and treats the other region as a fixed boundary.
    center_result = list(trajectory)
    upper_result = list(trajectory)
    initial_blank_geometry = _special_boundary_geometry(
        trajectory, target_spacing, mesh=mesh
    )
    if current_blank_centers is None:
        current_blank_centers = dict(
            initial_blank_geometry.get("blank_centers_3d", {})
        )
    else:
        current_blank_centers = dict(current_blank_centers)
    current_periodic_blocked = bool(
        initial_blank_geometry.get("periodic_pair_blocked", False)
    )
    step_kwargs = dict(kwargs)
    step_kwargs.update({
        "iterations": 1,
        # Each one-step regional pass must still evaluate the coverage field.
        # The final UPPER pass sees CENTER + LOWER as fixed spray references,
        # so its metric is the actual CV of the complete updated trajectory.
        "uniform_stop": True,
        "uniform_every": 1,
        "uniform_min_iterations": iterations + 1,
        "return_best_uniform": False,
    })
    red_motion_active = True
    # Keep the uniformity EMA across outer iterations.  Each regional pass
    # intentionally runs one step, so the inner optimizer cannot own this
    # state or every reported EMA would equal the current CV.
    outer_ema = None
    outer_ema_delta = float("nan")
    outer_ema_stable_count = 0
    outer_ema_patience = max(1, int(kwargs.get("uniform_patience", 4)))
    outer_ema_stop_requested = False

    for iteration in range(iterations):
        metric_state = {"energy": 0.0, "cv": float("nan"), "cv_smoothed": float("nan"),
                        "cv_threshold": float("nan"), "cv_stable": 0.0}
        outer_gap = float("inf")

        def capture(_preview, _it, _energy, components):
            metric_state["energy"] += float(_energy)
            metric_state.update(components)

        step_kwargs["blank_centers_3d"] = current_blank_centers
        step_kwargs["periodic_pair_blocked"] = current_periodic_blocked
        step_kwargs["on_iteration"] = capture
        center_result = _optimize_trajectory_region_energy(
            trajectory=upper_result,
            mesh=mesh,
            moving_region="CENTER",
            **step_kwargs,
        )
        if red_motion_active:
            lower_result = _optimize_trajectory_region_energy(
                trajectory=center_result,
                mesh=mesh,
                moving_region="LOWER",
                **step_kwargs,
            )
            upper_result = _optimize_trajectory_region_energy(
                trajectory=lower_result,
                mesh=mesh,
                moving_region="UPPER",
                **step_kwargs,
            )
        else:
            # Keep the red trajectories frozen while CENTER continues its
            # independent Aeigenopt-style coverage relaxation.
            upper_result = center_result

        latest_blank_geometry = _special_boundary_geometry(
            upper_result, target_spacing, mesh=mesh
        )
        # Eligibility controls whether a side may update its void center, not
        # whether an already established center still exists.  Keep the last
        # valid point when the live C0/C_last gaps later fall below 1.5d so
        # rendering and the next force step consume the same persistent state.
        latest_blank_centers = latest_blank_geometry.get(
            "blank_centers_3d", {}
        )
        if isinstance(latest_blank_centers, dict):
            for side in ("LOWER", "UPPER"):
                value = latest_blank_centers.get(side)
                if value is None:
                    continue
                point = np.asarray(value, dtype=float)
                if point.shape == (3,) and np.isfinite(point).all():
                    current_blank_centers[side] = point.tolist()
        current_periodic_blocked = bool(
            latest_blank_geometry.get("periodic_pair_blocked", False)
        )
        if on_iteration is not None:
            moved = []
            for before, after in zip(trajectory, upper_result):
                if (before.kind == "SURFACE_SCAN"
                        and str(getattr(before, "region", "")).upper() in {"CENTER", "LOWER", "UPPER"}
                        and len(before.xyz) > 0 and len(after.xyz) > 0):
                    before_tree = cKDTree(np.asarray(before.xyz, dtype=float))
                    moved.append(float(np.mean(before_tree.query(
                        np.asarray(after.xyz, dtype=float)
                    )[0])))
            motion_metric = float(np.mean(moved)) if moved else 0.0
            cv_value = float(metric_state.get(
                "uniform_metric_current",
                metric_state.get("uniform_metric", float("nan")),
            ))
            if np.isfinite(cv_value):
                previous_ema = outer_ema
                outer_ema = (
                    float(cv_value) if previous_ema is None
                    else 0.7 * float(previous_ema) + 0.3 * float(cv_value)
                )
                outer_ema_delta = (
                    float("nan") if previous_ema is None
                    else abs(float(outer_ema) - float(previous_ema))
                )
            cv_smoothed = float(
                outer_ema if outer_ema is not None else cv_value
            )
            # The outer-loop EMA is the displayed convergence signal.
            # Recompute its threshold from the same tolerances used by the
            # regional optimizer so the console values are directly useful.
            cv_threshold = (
                float(kwargs.get("uniform_abs_tol", 1e-6))
                + float(kwargs.get("uniform_rel_tol", 6.18e-4))
                * max(abs(cv_smoothed), 1.0e-12)
                if np.isfinite(cv_smoothed) else float("nan")
            )
            # Stop only after the EMA change remains below its threshold for
            # the configured number of consecutive outer iterations.
            if (
                    np.isfinite(outer_ema_delta)
                    and np.isfinite(cv_threshold)
                    and outer_ema_delta <= cv_threshold
            ):
                outer_ema_stable_count += 1
            else:
                outer_ema_stable_count = 0
            if outer_ema_stable_count >= outer_ema_patience:
                outer_ema_stop_requested = True
            on_iteration(
                upper_result,
                iteration + 1,
                float(metric_state.get("energy", 0.0)),
                {
                    "optimization_region": "CENTER->LOWER->UPPER",
                    "uniform_metric": cv_value if np.isfinite(cv_value) else motion_metric,
                    "uniform_metric_current": cv_value,
                    "uniform_metric_smoothed": cv_smoothed,
                    "uniform_delta": outer_ema_delta,
                    "uniform_threshold": cv_threshold,
                    "uniform_stable_count": float(outer_ema_stable_count),
                    "uniform_patience": float(outer_ema_patience),
                    "center_displacement_mean": motion_metric,
                    "center_displacement_max": motion_metric,
                    "outer_gap": float(outer_gap) if 'outer_gap' in locals() else float("inf"),
                    "red_motion_active": float(red_motion_active),
                    "blank_centers_3d": current_blank_centers,
                    "periodic_pair_blocked": current_periodic_blocked,
                },
                )

        if outer_ema_stop_requested:
            print(
                f"\n[Uniformity EMA stop] EMA_delta={outer_ema_delta:.6e} "
                f"<= EMA_stop={cv_threshold:.6e} for "
                f"{outer_ema_stable_count} consecutive iterations.",
                flush=True,
            )
            break

        # Once the outermost red samples are all within the target spacing of
        # CENTER, stop the coupled motion for every region together.
        center_segments = [s for s in upper_result
                           if s.kind == "SURFACE_SCAN"
                           and str(getattr(s, "region", "")).upper() == "CENTER"]
        red_segments = [s for index, s in enumerate(upper_result)
                        if s.kind == "SURFACE_SCAN"
                        and index in moving_red_indices]
        if center_segments and red_segments:
            center_points = np.vstack([
                np.asarray(s.xyz, dtype=float) for s in center_segments
                if len(s.xyz) > 0
            ])
            if len(center_points):
                center_tree = cKDTree(center_points)
                outer_gap = 0.0
                for segment in red_segments:
                    points = np.asarray(segment.xyz, dtype=float)
                    if len(points):
                        outer_gap = max(outer_gap,
                                        float(np.max(center_tree.query(points)[0])))
                if outer_gap <= target_spacing * 1.01:
                    print(f"[Energy optimizer] red boundary spacing reached d={target_spacing:.6g}; stopping.")
                    red_motion_active = False

    return upper_result


def optimize_trajectory_arcs_postprocess(
        trajectory: List[TrajectorySegment],
        mesh: trimesh.Trimesh,
        speed: float = 300.0,
        a_max: float = 1000.0,
        j_max: float = 5000.0,
        target_spacing: float = 30.0,
        sample_step: float = 1.0,
        center_boundary_limit: float | None = None,
        project_fillet_to_mesh: bool = False,
        arc_lock_guard_d: float = 2.0,
) -> List[TrajectorySegment]:
    v_data = extract_raw_v_trajectories(trajectory)
    superellipse_traj = fit_superellipse_trajectories(
        v_data,
        mesh=mesh,
        target_spacing=target_spacing,
        sample_step=sample_step,
    )
    final_traj = apply_tangent_circle_fillet(
        fitted_trajectory=superellipse_traj,
        mesh=mesh,
        speed=speed,
        a_max=a_max,
        j_max=j_max,
        target_spacing=target_spacing,
        sample_step=sample_step, # 确保把 sample_step 传给倒圆函数
        center_boundary_limit=center_boundary_limit,
        project_fillet_to_mesh=project_fillet_to_mesh,
        arc_lock_guard_d=arc_lock_guard_d,
    )
    return final_traj


def _trajectory_is_transition(segment: TrajectorySegment) -> bool:
    return (segment.kind in {"OVERTRAVEL", "SEAM_JUMP", "SEAM_BLEND_3D"}
            or str(getattr(segment, "region", "")).upper() == "TRANSITION"
            or not bool(segment.spray_on))


def _polyline_curvature(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    out = np.zeros(len(pts), dtype=float)
    if len(pts) < 3:
        return out
    a, b = pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1]
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (la > EPS) & (lb > EPS)
    if np.any(valid):
        ta, tb = a[valid] / la[valid, None], b[valid] / lb[valid, None]
        out[1:-1][valid] = np.arccos(np.clip(np.sum(ta * tb, axis=1), -1, 1)) / np.maximum((la[valid] + lb[valid]) * 0.5, EPS)
        out[0], out[-1] = out[1], out[-2]
    return out


def _adaptive_quintic_breaks(points: np.ndarray, sample_step: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 7:
        return np.array([0, n - 1], dtype=int)
    edge = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(edge))
    if total <= EPS:
        return np.array([0, n - 1], dtype=int)
    kappa = _polyline_curvature(pts)
    dk = np.abs(np.diff(kappa))
    k90 = float(np.percentile(kappa, 90)) if np.any(kappa > 0) else 0.0
    v90 = float(np.percentile(dk, 90)) if np.any(dk > 0) else 0.0
    baseline = max(1, int(np.ceil(total / max(20.0 * sample_step, 1.0e-3))))
    requested = max(1, int(np.ceil(baseline * (1.0 + 2.0 * np.clip(v90 / max(k90, EPS), 0, 1)))))
    count = min(requested, (n - 1) // 5)
    if count <= 1:
        return np.array([0, n - 1], dtype=int)
    weight = edge * (1.0 + 4.0 * (dk if len(dk) else np.zeros_like(edge)) / max(v90, EPS))
    cumulative = np.r_[0.0, np.cumsum(np.maximum(weight, EPS))]
    breaks = [0]
    for target in cumulative[-1] * np.arange(1, count, dtype=float) / count:
        idx = int(np.searchsorted(cumulative, target))
        idx = max(idx, breaks[-1] + 5)
        latest = n - 1 - (count - len(breaks) - 1) * 5
        breaks.append(min(idx, latest))
    breaks.append(n - 1)
    return np.asarray(sorted(set(breaks)), dtype=int)


def _quintic_fit_polyline(points: np.ndarray, sample_step: float) -> tuple[np.ndarray, int]:
    """Piecewise degree-five least-squares B-spline fit (not interpolation)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy(), 1
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1.0e-7]
    pts = pts[keep]
    if len(pts) < 2:
        return pts.copy(), 1
    breaks = _adaptive_quintic_breaks(pts, max(sample_step, 1.0e-3))
    pieces, used = [], 0
    for start, stop in zip(breaks[:-1], breaks[1:]):
        chunk = pts[int(start):int(stop) + 1]
        if len(chunk) < 6:
            # A degree-five fit is mathematically underdetermined here, but
            # the source samples must never be dropped (otherwise the first
            # render shows a local coverage gap).  Preserve this short piece.
            piece = chunk.copy()
            pieces.append(piece if not pieces else piece[1:])
            used += 1
            continue
        d = np.linalg.norm(np.diff(chunk, axis=0), axis=1)
        u = np.r_[0.0, np.cumsum(d)]
        if u[-1] <= EPS:
            continue
        u /= u[-1]
        # A small number of interior knots gives smoothing while retaining
        # enough degrees of freedom for the curvature-adaptive short pieces.
        n_internal = min(6, max(0, (len(chunk) - 6) // 4))
        knots = np.r_[[0.0] * 6, np.linspace(0.0, 1.0, n_internal + 2)[1:-1], [1.0] * 6]
        weights = np.ones(len(chunk), dtype=float)
        weights[[0, -1]] = 1.0e5  # lock C0 endpoints to the transition points
        try:
            spline = make_lsq_spline(u, chunk, knots, k=5, w=weights, axis=0)
            count = max(8, int(np.ceil(np.sum(d) / max(sample_step, 1.0e-3))) + 1)
            piece = np.asarray(spline(np.linspace(0.0, 1.0, count)), dtype=float)
        except Exception:
            # Never discard a failed adaptive interval.  Keeping its original
            # samples is preferable to creating a hole in the spray path.
            piece = chunk.copy()
        piece[0], piece[-1] = chunk[0], chunk[-1]
        pieces.append(piece if not pieces else piece[1:])
        used += 1
    return (np.vstack(pieces), used) if pieces else (pts.copy(), 0)


def fit_quintic_bspline_trajectories(
        trajectory: List[TrajectorySegment], mesh: trimesh.Trimesh,
        speed: float, a_max: float, j_max: float, sample_step: float = 1.0,
        target_spacing: float = 30.0, tol_pct: float = 10.0,
        overtravel_factor: float = 2.0, project_to_mesh: bool = True,
) -> List[TrajectorySegment]:
    """Fit every non-transition trajectory with adaptive quintic splines."""
    _ = speed, a_max, j_max, target_spacing, tol_pct, overtravel_factor
    result: List[TrajectorySegment] = []
    for seg in trajectory:
        xyz = np.asarray(seg.xyz, dtype=float)
        if _trajectory_is_transition(seg) or len(xyz) < 2:
            result.append(seg)
            continue
        fitted, pieces = _quintic_fit_polyline(xyz, sample_step)
        if pieces <= 0:
            result.append(seg)
            continue
        if project_to_mesh:
            p0, p1 = fitted[0].copy(), fitted[-1].copy()
            fitted, _, _ = trimesh.proximity.closest_point(mesh, fitted)
            fitted[0], fitted[-1] = p0, p1
        meta = dict(seg.geometry_meta) if isinstance(seg.geometry_meta, dict) else {}
        meta.update({"quintic_degree": 5, "quintic_pieces": pieces, "fit_type": "least_squares"})
        result.append(TrajectorySegment(seg.kind, np.full((len(fitted), 2), np.nan), fitted,
                                        seg.spray_on, seg.note.split(" [5th")[0] +
                                        f" [5th B-Spline least-squares, pieces={pieces}]",
                                        seg.region, np.where(np.arange(len(fitted)) < len(fitted) // 2, 0, 1), meta))
    return result


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
    lower_left_candidates = _available_tracks(
        lower_left_x[::-1], "lower", local_uv, data.uv_faces, tolerance
    )
    lower_right_candidates = _available_tracks(
        lower_right_x[::-1], "lower", local_uv, data.uv_faces, tolerance
    )
    upper_left_candidates = _available_tracks(
        upper_left_x, "upper", local_uv, data.uv_faces, tolerance
    )
    upper_right_candidates = _available_tracks(
        upper_right_x, "upper", local_uv, data.uv_faces, tolerance
    )

    # Reject the tiny terminal strips visible at the tapered boundary.  The
    # 2*d criterion removes the v4Hda tip pass (about 45--50 mm UV span) while
    # retaining the preceding full-width pass.  Apply the same rule to both
    # sides so one-to-one pairing remains aligned.
    minimum_outer_span = 2.0 * target_spacing
    lower_left, rejected_lower_boundary_left = _filter_short_boundary_tracks(
        lower_left_candidates, minimum_outer_span
    )
    lower_right, rejected_lower_boundary_right = _filter_short_boundary_tracks(
        lower_right_candidates, minimum_outer_span
    )
    upper_left, rejected_upper_boundary_left = _filter_short_boundary_tracks(
        upper_left_candidates, minimum_outer_span
    )
    upper_right, rejected_upper_boundary_right = _filter_short_boundary_tracks(
        upper_right_candidates, minimum_outer_span
    )

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

    # =========================================================================
    # 4. 下半区排序与共轭连线（基于完美 1-to-1 共轭拓扑）
    # =========================================================================
    paired_lower = min(len(lower_left), len(lower_right))
    lower_first_is_left: bool | None = None
    for index in range(paired_lower):
        left_track_data = lower_left[index]
        right_track_data = lower_right[index]

        # 探路判定首条轨迹的启动侧，就近连接上一个表面终点
        left_probe = _make_surface_scan(*left_track_data, True, sample_step, frame, mapper)
        right_probe = _make_surface_scan(*right_track_data, True, sample_step, frame, mapper)

        if last_surface is not None and lower_first_is_left is None:
            left_gap = float(np.linalg.norm(last_surface.xyz[-1] - left_probe.xyz[0]))
            right_gap = float(np.linalg.norm(last_surface.xyz[-1] - right_probe.xyz[0]))
            lower_first_is_left = (left_gap <= right_gap)
        else:
            # Continue a deterministic serpentine path: alternate sides after
            # the initial nearest-endpoint choice instead of re-solving the
            # side independently for every pair.
            if lower_first_is_left is None:
                lower_first_is_left = True
        first_is_left = lower_first_is_left if index % 2 == 0 else not lower_first_is_left

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
    upper_first_is_left: bool | None = None
    for index in range(paired_upper):
        left_track_data = upper_left[index]
        right_track_data = upper_right[index]

        # 探路判定首条轨迹启动侧，优先连接中间区最后一个终点
        left_probe = _make_surface_scan(*left_track_data, False, sample_step, frame, mapper)
        right_probe = _make_surface_scan(*right_track_data, False, sample_step, frame, mapper)

        if last_surface is not None and upper_first_is_left is None:
            left_gap = float(np.linalg.norm(last_surface.xyz[-1] - left_probe.xyz[0]))
            right_gap = float(np.linalg.norm(last_surface.xyz[-1] - right_probe.xyz[0]))
            upper_first_is_left = (left_gap <= right_gap)
        else:
            if upper_first_is_left is None:
                upper_first_is_left = True
        first_is_left = upper_first_is_left if index % 2 == 0 else not upper_first_is_left

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
            "lower_rejected_short_boundary": (
                rejected_lower_boundary_left + rejected_lower_boundary_right
            ),
            "upper_rejected_short_boundary": (
                rejected_upper_boundary_left + rejected_upper_boundary_right
            ),
            "lower_rejected_spacing": rejected_lower_spacing_tracks,
            "upper_rejected_spacing": rejected_upper_spacing_tracks,
        },
        "segment_count": len(segments),
    }
    metadata["special_boundary_geometry"] = _special_boundary_geometry(
        segments,
        target_spacing,
        mesh=trimesh.Trimesh(vertices=data.xyz, faces=data.xyz_faces, process=False),
    )
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
    point_segments = [
        (segment, xyz)
        for segment in segments
        if len(segment.xyz)
        for xyz in segment.xyz
    ]
    if not point_segments:
        path.write_text("", encoding="utf-8")
        return

    surface_points = np.asarray([item[1] for item in point_segments], dtype=float)
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, surface_points)
    normals = np.asarray(mesh.face_normals, dtype=float)[np.asarray(face_ids, dtype=np.int64)]
    inward_normals = -normals

    # 位置点偏置使用指向外部的正常面法向 normals
    # Keep free-space transition points in free space.  Only spray-on surface
    # points receive projection and spray-distance offset.
    export_points = surface_points.copy()
    surface_mask = np.asarray(
        [segment.kind == "SURFACE_SCAN" for segment, _ in point_segments],
        dtype=bool,
    )
    # Keep the fitted trajectory itself, especially its exact open-boundary
    # endpoints.  Replacing every point by closest-point coordinates can pull
    # a valid boundary point back into the mesh.  Only apply the requested gun
    # offset along the normal.
    export_points[surface_mask] = (
        surface_points[surface_mask] + normals[surface_mask] * float(spray_distance)
    )

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
    actor.GetProperty().SetRenderLinesAsTubes(False)

    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
    return actor

# def vtk_actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float], opacity: float = 0.90,
#               width: float = 1.0, wireframe: bool = False) -> vtk.vtkActor:
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(polydata)
#     mapper.SetResolveCoincidentTopologyToPolygonOffset()
#
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetColor(*color)
#     actor.GetProperty().SetOpacity(opacity)
#     actor.GetProperty().SetLineWidth(width)
#     actor.GetProperty().SetRenderLinesAsTubes(False)  # 改为 False，消除几何生成开销
#
#     if wireframe:
#         actor.GetProperty().SetRepresentationToWireframe()
#     return actor


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
    """Draw the CV history used by the live optimizer's stop criterion."""
    table = vtk.vtkTable()
    iteration_values = vtk.vtkDoubleArray()
    energy_values = vtk.vtkDoubleArray()
    iteration_values.SetName("iteration")
    energy_values.SetName("smoothed coefficient of variation")
    for iteration, energy in zip(iterations, energies):
        iteration_values.InsertNextValue(float(iteration))
        energy_values.InsertNextValue(float(energy))
    table.AddColumn(iteration_values)
    table.AddColumn(energy_values)

    chart = vtk.vtkChartXY()
    latest = float(energies[-1])
    chart.SetTitle(f"Uniformity stop metric: CV={latest:.6g}")
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
    y_axis.SetTitle("smoothed coefficient of variation (CV)")
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
        self.optimization_reference: list[TrajectorySegment] = []
        self.optimization_events: queue.Queue = queue.Queue()
        self.optimization_generation = 0
        self.current_metadata: dict[str, object] | None = None
        self.a_max = 1000.0  # 默认值
        self.j_max = 5000.0  # 默认值
        self.hide_model = False
        self.hide_traj = False

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("OptCuts trajectory - VTK/OpenGL (Agien)")
        self.window.SetSize(1600, 900)
        self.window.SetMultiSamples(0)
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
        # Keep the base layer visible whenever the trajectory overlay redraws.
        self.left_overlay_renderer.PreserveColorBufferOn()
        self.left_overlay_renderer.PreserveDepthBufferOff()
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
        self.interactor.SetDesiredUpdateRate(30.0)
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
        fixed_scan_append_3d = vtk.vtkAppendPolyData()
        center_scan_append_3d = vtk.vtkAppendPolyData()
        overtravel_append_3d = vtk.vtkAppendPolyData()
        blend_append_3d = vtk.vtkAppendPolyData()

        has_fixed_scan = False
        has_center_scan = False
        has_overtravel = False
        has_blend = False

        for seg in trajectory:
            if len(seg.xyz) < 2:
                continue
            poly = vtk_polyline(seg.xyz)
            if seg.kind == "SURFACE_SCAN":
                if str(getattr(seg, "region", "")).upper() == "CENTER":
                    center_scan_append_3d.AddInputData(poly)
                    has_center_scan = True
                else:
                    fixed_scan_append_3d.AddInputData(poly)
                    has_fixed_scan = True
            elif seg.kind == "OVERTRAVEL":
                overtravel_append_3d.AddInputData(poly)
                has_overtravel = True
            elif seg.kind == "SEAM_BLEND_3D":
                blend_append_3d.AddInputData(poly)
                has_blend = True

        # 绘制主喷涂轨迹 (红色)
        if has_fixed_scan:
            fixed_scan_append_3d.Update()
            actor = vtk_actor(fixed_scan_append_3d.GetOutput(), (0.72, 0.25, 0.28), opacity=0.55, width=2.5)
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        if self.optimization_reference:
            reference_append = vtk.vtkAppendPolyData()
            for seg in self.optimization_reference:
                if seg.kind == "SURFACE_SCAN" and len(seg.xyz) >= 2:
                    reference_append.AddInputData(vtk_polyline(seg.xyz))
            reference_append.Update()
            if reference_append.GetOutput().GetNumberOfLines() > 0:
                actor = vtk_actor(reference_append.GetOutput(), (0.20, 0.24, 0.30), opacity=0.42, width=1.5)
                self.left_overlay_renderer.AddActor(actor)
                self.trajectory_actors_3d.append(actor)

        if has_center_scan:
            center_scan_append_3d.Update()
            actor = vtk_actor(center_scan_append_3d.GetOutput(), (1.0, 0.72, 0.05), opacity=1.0, width=4.5)
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

        # Periodic center-boundary connectors: blue, deliberately rendered on
        # top of the mesh so the 0/360-degree neighbour relationship is clear.
        # Recompute from the displayed (possibly optimized) trajectory so the
        # blue geometry follows live edits rather than the initial scan set.
        special_geometry = _special_boundary_geometry(
            trajectory,
            float(metadata.get("optimized_spacing", self.sample_step)),
            mesh=self.mesh,
        )
        current_blank_centers = metadata.get("optimization_blank_centers_3d")
        if isinstance(current_blank_centers, dict):
            special_geometry["blank_centers_3d"] = dict(current_blank_centers)
        metadata["special_boundary_geometry"] = special_geometry
        # All candidate 1.5d connectors remain blue.
        connector_3d = (
            special_geometry.get("connectors_3d", [])
            if isinstance(special_geometry, dict)
            else []
        )
        if connector_3d:
            append = vtk.vtkAppendPolyData()
            for line in connector_3d:
                points = np.asarray(line, dtype=float)
                if (
                    points.shape == (2, 3)
                    and np.linalg.norm(points[1] - points[0]) > EPS
                ):
                    append.AddInputData(vtk_polyline(points))
            append.Update()
            if append.GetOutput().GetNumberOfLines() > 0:
                actor = vtk_actor(
                    append.GetOutput(),
                    (0.05, 0.25, 1.0),
                    opacity=0.35,
                    width=2.0,
                )
                self.left_overlay_renderer.AddActor(actor)
                self.trajectory_actors_3d.append(actor)

        # Only the selected periodic/FIRST/LAST connectors are green.
        selected_3d = []
        selected_records = (
            special_geometry.get("selected_connector_records", {})
            if isinstance(special_geometry, dict)
            else {}
        )
        seen_selected = set()

        if isinstance(selected_records, dict):
            for records in selected_records.values():
                if not isinstance(records, list):
                    continue
                for record in records:
                    line = record.get("line_3d")
                    if line is None:
                        continue
                    points = np.asarray(line, dtype=float)
                    if points.shape != (2, 3):
                        continue
                    if np.linalg.norm(points[1] - points[0]) <= EPS:
                        continue

                    key = tuple(np.round(points.reshape(-1), 8))
                    if key in seen_selected:
                        continue
                    seen_selected.add(key)
                    selected_3d.append(points)

        if selected_3d:
            append = vtk.vtkAppendPolyData()
            for points in selected_3d:
                append.AddInputData(vtk_polyline(points))
            append.Update()
            if append.GetOutput().GetNumberOfLines() > 0:
                actor = vtk_actor(
                    append.GetOutput(),
                    (0.0, 0.85, 0.1),
                    opacity=1.0,
                    width=5.0,
                )
                self.left_overlay_renderer.AddActor(actor)
                self.trajectory_actors_3d.append(actor)
        blank_centers = special_geometry.get("blank_centers_3d", {})
        if isinstance(blank_centers, dict) and blank_centers:
            points = np.asarray(list(blank_centers.values()), dtype=float)
            actor = vtk_actor(vtk_points(points), (0.05, 0.95, 0.15), opacity=1.0)
            actor.GetProperty().SetPointSize(16.0)
            try:
                actor.GetProperty().RenderPointsAsSpheresOn()
            except AttributeError:
                pass
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        repulsion_centers = special_geometry.get("repulsion_centers_3d", {})
        if isinstance(repulsion_centers, dict) and repulsion_centers:
            points = np.asarray(list(repulsion_centers.values()), dtype=float)
            actor = vtk_actor(vtk_points(points), (1.0, 0.02, 0.02), opacity=1.0)
            actor.GetProperty().SetPointSize(16.0)
            try:
                actor.GetProperty().RenderPointsAsSpheresOn()
            except AttributeError:
                pass
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

        connector_2d = special_geometry.get("connectors_2d", []) if isinstance(special_geometry, dict) else []
        for line in connector_2d:
            points = np.asarray(line, dtype=float)
            if points.shape == (2, 2) and np.linalg.norm(points[1] - points[0]) > EPS:
                actor_2d = vtk_actor(vtk_polyline(points), (0.05, 0.25, 1.0), opacity=1.0, width=4.0)
                self.right_renderer.AddActor(actor_2d)
                self.trajectory_actors_2d.append(actor_2d)

        # Highlight only the selected periodic/FIRST/LAST or MODEL_BOUNDARY
        # connectors in green on the UV view as well.
        selected_2d = []
        seen_selected_2d = set()
        if isinstance(selected_records, dict):
            for records in selected_records.values():
                if not isinstance(records, list):
                    continue
                for record in records:
                    line = record.get("line_2d")
                    if line is None:
                        continue
                    points = np.asarray(line, dtype=float)
                    if points.shape != (2, 2):
                        continue
                    if np.linalg.norm(points[1] - points[0]) <= EPS:
                        continue
                    key = tuple(np.round(points.reshape(-1), 8))
                    if key in seen_selected_2d:
                        continue
                    seen_selected_2d.add(key)
                    selected_2d.append(points)

        for points in selected_2d:
            actor_2d = vtk_actor(
                vtk_polyline(points),
                (0.0, 0.85, 0.1),
                opacity=1.0,
                width=5.0,
            )
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
        self.root.geometry("400x380+50+50")
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

        tk.Label(self.root, text="Arc lock guard (total d): ", font=("Arial", 11)).grid(
            row=5, column=0, padx=10, pady=6, sticky="w"
        )
        self.arc_lock_guard_var = tk.StringVar(value="2.0")
        self.arc_lock_guard_entry = tk.Entry(
            self.root, textvariable=self.arc_lock_guard_var, width=15, font=("Arial", 11)
        )
        self.arc_lock_guard_entry.grid(row=5, column=1, padx=10, pady=6, sticky="ew")

        # 原有的按钮行号下移
        self.btn = tk.Button(self.root, text="更新轨迹", command=self.update_spacing, font=("Arial", 11), bg="#1948CD",
                             fg="white")
        self.btn.grid(row=6, column=0, padx=10, pady=5, sticky="ew")

        self.export_btn = tk.Button(self.root, text="导出轨迹", command=self.export_trajectory, font=("Arial", 11),
                                    bg="#1948CD", fg="white")
        self.export_btn.grid(row=6, column=1, padx=10, pady=5, sticky="ew")

        # Show/Hide checkboxes
        self.hide_model_var = tk.BooleanVar(value=False)
        self.cb_model = tk.Checkbutton(
            self.root, text="隐藏三维网格模型", variable=self.hide_model_var, command=self.toggle_model,
            font=("Arial", 10)
        )
        self.cb_model.grid(row=7, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        self.hide_traj_var = tk.BooleanVar(value=False)
        self.cb_traj = tk.Checkbutton(
            self.root, text="隐藏喷涂加工轨迹", variable=self.hide_traj_var, command=self.toggle_traj,
            font=("Arial", 10)
        )
        self.cb_traj.grid(row=8, column=0, columnspan=2, sticky="w", padx=10, pady=2)

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
            a_max = float(self.a_max_var.get())
            j_max = float(self.j_max_var.get())
            arc_lock_guard_d = float(self.arc_lock_guard_var.get())

            if not np.isfinite(new_d) or new_d <= 0: raise ValueError
            if not np.isfinite(speed) or speed <= 0: raise ValueError
            if not np.isfinite(a_max) or a_max <= 0: raise ValueError
            if not np.isfinite(j_max) or j_max <= 0: raise ValueError
            if not np.isfinite(arc_lock_guard_d) or arc_lock_guard_d < 0: raise ValueError

        except ValueError:
            messagebox.showerror("错误", "参数必须为大于 0 的有效数字。")
            return

        self.visualizer.current_spacing = new_d
        self.visualizer.trajectory_speed = speed
        self.visualizer.spray_distance = spray_distance
        self.visualizer.a_max = a_max
        self.visualizer.j_max = j_max

        # 动力学参数打印
        R_calc = max((speed ** 2) / a_max, math.sqrt((speed ** 3) / j_max))
        print(
            f"\n[动力学参数触发] v={speed:.1f}mm/s, a_max={a_max:.1f}, j_max={j_max:.1f} => 理论最小约束半径 R_min={R_calc:.2f}mm")

        # 生成初代规划轨迹
        raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
            self.visualizer.data,
            self.visualizer.precut_seam,
            target_spacing=new_d,
            sample_step=self.visualizer.sample_step,
        )

        # 1. 提取 V 形特征
        v_data = extract_raw_v_trajectories(raw_trajectory)

        # 2. 提前 d 截断并拟合超椭圆
        superellipse_traj = fit_superellipse_trajectories(
            v_data,
            mesh=self.visualizer.mesh,  # main 中使用 mesh=mesh
            target_spacing=new_d,  # main 中使用 target_spacing=selected_spacing
            sample_step=self.visualizer.sample_step
        )

        # 3. 动力学工况 A/B 倒圆并更新吸附
        optimized = apply_tangent_circle_fillet(
            fitted_trajectory=superellipse_traj,
            mesh=self.visualizer.mesh,  # main 中使用 mesh=mesh
            speed=speed,  # main 中使用 speed=args.trajectory_speed
            a_max=a_max,  # main 中使用 a_max=selected_a_max
            j_max=j_max,  # main 中使用 j_max=selected_j_max
            target_spacing=new_d,  # main 中使用 target_spacing=selected_spacing
            sample_step=self.visualizer.sample_step,
            arc_lock_guard_d=arc_lock_guard_d,
        )

        fixed_blank_geometry = _special_boundary_geometry(
            optimized, new_d, mesh=self.visualizer.mesh
        )
        metadata["optimization_blank_centers_3d"] = dict(
            fixed_blank_geometry.get("blank_centers_3d", {})
        )

        # Relax the coupled straight portions first; quintic fitting is done
        # once from the completed relaxed geometry in the worker below.
        self.visualizer.reset_energy_history()
        self.visualizer.optimization_reference = list(optimized)
        self.visualizer.optimization_generation += 1
        generation = self.visualizer.optimization_generation

        def enqueue_iteration(preview, iteration, total_energy, energy_components):
            self.visualizer.optimization_events.put((
                generation, "iteration", preview, iteration, total_energy,
                energy_components, planning_frame, metadata, raw_trajectory,
            ))

        def run_optimization():
            try:
                result = optimize_center_trajectories_dual_blank_energy(
                    trajectory=optimized, mesh=self.visualizer.mesh,
                    target_spacing=new_d, sample_step=self.visualizer.sample_step,
                    blank_centers_3d=metadata["optimization_blank_centers_3d"],
                    on_iteration=enqueue_iteration,
                )
                result = fit_quintic_bspline_trajectories(
                    trajectory=result, mesh=self.visualizer.mesh,
                    speed=speed, a_max=a_max, j_max=j_max,
                    sample_step=self.visualizer.sample_step,
                    target_spacing=new_d, project_to_mesh=False,
                )
                self.visualizer.optimization_events.put((
                    generation, "complete", result, planning_frame, metadata,
                    raw_trajectory,
                ))
            except Exception as exc:
                self.visualizer.optimization_events.put((generation, "error", exc))

        threading.Thread(
            target=run_optimization, name="center-trajectory-optimizer", daemon=True
        ).start()
        self.visualizer.update_trajectory(
            optimized, planning_frame, metadata, trajectory_2d=raw_trajectory
        )
        self.visualizer.window.Render()
        print(f"成功将目标间距 d 更新为: {new_d:.6g}，并完成高保真超椭圆拟合！")

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
        now = time.monotonic()
        last = getattr(tk_panel, "_last_event_pump", 0.0)
        if now - last < 0.05:
            return
        tk_panel._last_event_pump = now
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
    """Run the interactive visualizer and display live spline-boundary optimization."""
    metadata = dict(metadata)
    fixed_blank_geometry = _special_boundary_geometry(
        trajectory, initial_spacing, mesh=mesh
    )
    metadata["optimization_blank_centers_3d"] = dict(
        fixed_blank_geometry.get("blank_centers_3d", {})
    )
    visualizer = InteractiveVisualizer(
        mesh, data, precut_seam, cut_edges, sample_step, initial_spacing,
        csv_path, metadata_path, paq_path, trajectory_speed, spray_distance
    )
    # 绘制传入的超椭圆拟合后轨迹
    visualizer.update_trajectory(trajectory, planning_frame, metadata, trajectory_2d=trajectory_2d)
    visualizer.reset_cameras()
    panel = TkControlPanel(visualizer)


    # 1. 先渲染出 VTK 窗口
    visualizer.window.Render()
    visualizer.interactor.Initialize()

    print("\n[开始实时样条边界能量场轨迹优化...]")
    visualizer.reset_energy_history()
    visualizer.optimization_reference = list(trajectory)

    visualizer.optimization_generation += 1
    generation = visualizer.optimization_generation
    optimization_events = visualizer.optimization_events

    def live_iteration_callback(preview, iteration, total_energy, energy_components):
        optimization_events.put((
            generation, "iteration", preview, iteration, total_energy,
            energy_components, planning_frame, metadata, trajectory_2d,
        ))

    def run_optimization():
        try:
            print(f"[Energy worker started] task={generation}", flush=True)
            optimized = optimize_center_trajectories_dual_blank_energy(
                trajectory=list(trajectory),
                mesh=mesh,
                target_spacing=initial_spacing,
                sample_step=sample_step,
                blank_centers_3d=metadata["optimization_blank_centers_3d"],
                on_iteration=live_iteration_callback,
            )
            optimized = fit_quintic_bspline_trajectories(
                trajectory=optimized,
                mesh=mesh,
                speed=trajectory_speed,
                a_max=1000.0,
                j_max=5000.0,
                sample_step=sample_step,
                target_spacing=initial_spacing,
                project_to_mesh=False,
            )
            optimization_events.put((
                generation, "complete", optimized, planning_frame, metadata,
                trajectory_2d,
            ))
        except Exception as exc:
            import traceback
            error_traceback = traceback.format_exc()
            print(
                f"[Energy worker failed] task={generation}: {exc!r}\n"
                f"{error_traceback}",
                flush=True,
            )
            optimization_events.put((generation, "error", exc, error_traceback))
    print("\n[轨迹准备就绪，开启可视化界面！]")
    # 进入主事件循环
    def on_timer(obj, event):
        tick_tkinter(obj, event, panel)
        for _ in range(2):
            try:
                update = optimization_events.get_nowait()
            except queue.Empty:
                break
            if update[0] != visualizer.optimization_generation:
                continue
            if update[1] == "iteration":
                (_, _, preview, iteration, total_energy, components,
                 event_frame, event_metadata, event_trajectory_2d) = update
                stop_metric = float(components.get(
                    "uniform_metric_smoothed",
                    components.get("uniform_metric", total_energy),
                ))
                latest_blank_centers = components.get("blank_centers_3d")
                if isinstance(latest_blank_centers, dict):
                    event_metadata["optimization_blank_centers_3d"] = dict(
                        latest_blank_centers
                    )
                visualizer.record_energy(iteration, stop_metric)
                visualizer.update_trajectory(
                    preview, event_frame, event_metadata,
                    trajectory_2d=event_trajectory_2d,
                )
                print(
                    f"\rOptimization {iteration}: E={total_energy:.6e} "
                    f"CV={components.get('uniform_metric_current', float('nan')):.6e} "
                    f"EMA={components.get('uniform_metric_smoothed', float('nan')):.6e} "
                    f"EMA_delta={components.get('uniform_delta', float('nan')):.3e} "
                    f"EMA_delta<=threshold({components.get('uniform_threshold', float('nan')):.3e}) "
                    f"={'YES' if (np.isfinite(components.get('uniform_delta', float('nan'))) and np.isfinite(components.get('uniform_threshold', float('nan'))) and components.get('uniform_delta', float('inf')) <= components.get('uniform_threshold', float('-inf'))) else 'NO'}; "
                    f"stable={int(components.get('uniform_stable_count', 0))}/"
                    f"{int(components.get('uniform_patience', 0))} "
                    f"move(mean/max)={components['center_displacement_mean']:.4f}/"
                    f"{components['center_displacement_max']:.4f} mm",
                    end="", flush=True,
                )
            elif update[1] == "complete":
                _, _, optimized, event_frame, event_metadata, event_trajectory_2d = update
                visualizer.update_trajectory(
                    optimized, event_frame, event_metadata,
                    trajectory_2d=event_trajectory_2d,
                )
                visualizer.current_trajectory = optimized
                print("\n[Live optimization complete]")
            else:
                print(f"\n[Live optimization failed] {update[2]!r}")
                if len(update) > 3:
                    print(update[3], flush=True)
        visualizer.window.Render()

    visualizer.interactor.AddObserver("TimerEvent", on_timer)
    visualizer.interactor.CreateRepeatingTimer(16)
    threading.Thread(
        target=run_optimization, name="center-trajectory-optimizer", daemon=True
    ).start()
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
    parser.add_argument(
        "--arc-center-limit", type=float, default=0.0,
        help="Maximum mesh distance of a B-arc center; 0 uses the empirical R+2d limit.",
    )
    parser.add_argument(
        "--arc-lock-guard-d", type=float, default=2.0,
        help="Total straight-arm guard locked around each arc, in multiples of d (default: 2).",
    )

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
    if not np.isfinite(args.arc_lock_guard_d) or args.arc_lock_guard_d < 0.0:
        raise ValueError("--arc-lock-guard-d must be finite and non-negative.")
    selected = args.mesh or _select_mesh_file()
    if not selected:
        return
    selected_path = Path(selected).resolve()
    if selected_path.suffix.lower() == ".json":
        mesh_path, optcuts_mesh, result_obj, precut_seam, data, optimized_cut_edges = _load_cutopt_cache(selected_path)
        mesh = _load_mesh(str(mesh_path))
        selected_path = mesh_path
        print(f"Loaded OptCuts cache: {selected_path}")
    else:
        mesh = _load_mesh(str(selected_path))
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
        _save_cutopt_cache(selected_path, official_input, result_obj, precut_seam, optimized_cut_edges)
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
        selected_spacing, selected_a_max, selected_j_max = _request_spacing(
            mesh, data, precut_seam, cut_edges, args.spacing,
            initial_a_max=args.a_max,
            initial_j_max=args.j_max,
        )

    if selected_spacing is None:
        print("Trajectory generation cancelled.")
        return
    sample_step_val = selected_spacing / 4.0 if args.sample_step <= 0.0 else args.sample_step
    raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
        data, precut_seam, target_spacing=selected_spacing, sample_step=sample_step_val
    )

    # 统一后处理入口：截断 + 超椭圆拟合 + 动力学倒圆
    arc_trajectory = optimize_trajectory_arcs_postprocess(
        trajectory=raw_trajectory,
        mesh=mesh,
        speed=args.trajectory_speed,
        a_max=selected_a_max,
        j_max=selected_j_max,
        target_spacing=selected_spacing,
        sample_step=sample_step_val,
        center_boundary_limit=(args.arc_center_limit if args.arc_center_limit > 0.0 else None),
        arc_lock_guard_d=args.arc_lock_guard_d,
    )

    # 对优化后的每条喷涂轨迹做曲率自适应五次最小二乘拟合；过渡段保持原样。
    if args.no_show:
        relaxed_trajectory = optimize_center_trajectories_dual_blank_energy(
        trajectory=arc_trajectory,
        mesh=mesh,
        sample_step=sample_step_val,
        target_spacing=selected_spacing,
        # optimize_trajectory_arcs_postprocess 已完成必要的曲面投影；
        # 再对整条拟合曲线做 closest-point 投影会在开口边界处把尾部拉回内部。
        )
        final_trajectory = fit_quintic_bspline_trajectories(
        trajectory=relaxed_trajectory,
        mesh=mesh,
        speed=args.trajectory_speed,
        a_max=selected_a_max,
        j_max=selected_j_max,
        sample_step=sample_step_val,
        target_spacing=selected_spacing,
            project_to_mesh=False,
        )
    else:
        # _show_result starts this same relaxation in its worker so VTK can
        # appear immediately instead of waiting for all optimization rounds.
        final_trajectory = arc_trajectory

    csv_path = Path(args.trajectory_out) if args.trajectory_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.csv"
    )
    metadata_path = Path(args.metadata_out) if args.metadata_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.json"
    )
    paq_path = Path(args.paq_out) if args.paq_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory_PAQ.txt"
    )

    _export_paq_trajectory(paq_path, mesh, final_trajectory,
                           args.trajectory_speed, args.spray_distance)

    print(f"Pre-cut shortest-boundary seam edges: {len(precut_seam.edges)}")
    print(f"Additional detected OptCuts seam edges: {len(optimized_cut_edges)}")
    print(f"Trajectory events: {len(final_trajectory)}")

    if not args.no_show:
        _show_result(
            mesh, data, precut_seam, cut_edges, final_trajectory, planning_frame, metadata,
            selected_spacing, sample_step_val, csv_path, metadata_path,
            paq_path, args.trajectory_speed, args.spray_distance,
            trajectory_2d=raw_trajectory,
        )
    if not args.no_animation:
        _play_iteration_animation(optcuts_mesh, result_obj)


if __name__ == "__main__":
    main()
