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
import cutopt_cache
from typing import Iterable, List, Optional, Sequence, Tuple
import heapq
import numpy as np
import trimesh
import vtk
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
#\(\text{原始轨迹}这个有V尖角截断，然后拟合成B样条
ROOT = Path(__file__).resolve().parents[2]
OPTCUTS_ROOT = ROOT / "third_party" / "OptCuts"
OPTCUTS_EXE = OPTCUTS_ROOT / "build" / "Release" / "OptCuts_bin.exe"
FINAL_FRAME_HOLD_MS = 2000
DEFAULT_TRAJECTORY_SPACING = 30.0
DEFAULT_TRAJECTORY_SPEED = 50.0
DEFAULT_SPRAY_DISTANCE = 0.0
EPS = 1.0e-9


@dataclass(frozen=True)
class PrecutSeam:
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
    physical_index: int
    uv_a0: np.ndarray
    uv_a1: np.ndarray
    uv_b0: np.ndarray
    uv_b1: np.ndarray


@dataclass
class TrajectorySegment:
    kind: str
    uv: np.ndarray
    xyz: np.ndarray
    spray_on: bool
    note: str = ""
    region: str = ""


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

    # 建立切割参考平面
    upper_pca_center, upper_pca_axes, _ = compute_pca(top.points)
    green_axis = _unit(upper_pca_axes[1], "upper-boundary PCA green axis")
    third_point = upper_pca_center + green_axis
    plane = build_reference_plane(upper_pca_center, bottom.center, third_point)

    curves = [
        np.asarray(curve, dtype=float)
        for curve in intersect_mesh_with_plane(mesh, plane)
        if len(curve) >= 2 and _polyline_length(curve) > EPS
    ]
    if len(curves) < 2:
        raise ValueError(
            "The loft three-point plane produced fewer than two usable intersections."
        )

    main_curves = sorted(curves, key=_polyline_length, reverse=True)[:2]
    curve_lengths = [_polyline_length(curve) for curve in main_curves]
    selected_index = int(np.argmin(curve_lengths))
    selected = main_curves[selected_index]

    vertices = np.asarray(mesh.vertices, dtype=float)
    bottom_ids = np.asarray(bottom.vertex_indices, dtype=np.int64)
    top_ids = np.asarray(top.vertex_indices, dtype=np.int64)

    forward_cost = np.min(
        np.linalg.norm(vertices[bottom_ids] - selected[0], axis=1)
    ) + np.min(np.linalg.norm(vertices[top_ids] - selected[-1], axis=1))
    reverse_cost = np.min(
        np.linalg.norm(vertices[bottom_ids] - selected[-1], axis=1)
    ) + np.min(np.linalg.norm(vertices[top_ids] - selected[0], axis=1))
    if reverse_cost < forward_cost:
        selected = selected[::-1].copy()

    # 统计网格尺寸与尺度自适应容差
    edges = mesh_edges_from_faces(faces)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    typical_edge = float(np.median(edge_lengths)) if len(edge_lengths) > 0 else 1.0

    model_scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    adaptive_eps = max(1e-7 * model_scale, 1e-6)

    dense_curve = _dense_polyline_samples(selected, 0.2 * typical_edge)
    tree = cKDTree(dense_curve)
    tri_centers = vertices[faces].mean(axis=1)
    dists, _ = tree.query(tri_centers)

    new_vertices = list(vertices)
    new_faces = []
    edge_cut_cache = {}
    plane_edges = set()

    bottom_edges_set = {
        (min(a, b), max(a, b)) for a, b in zip(bottom_ids, np.roll(bottom_ids, -1))
    }
    top_edges_set = {
        (min(a, b), max(a, b)) for a, b in zip(top_ids, np.roll(top_ids, -1))
    }

    new_bottom_vertices = set(bottom_ids)
    new_top_vertices = set(top_ids)

    # 采用尺度自适应容差清零距离
    s_all = vertices @ plane.normal + plane.d
    s_all[np.abs(s_all) < adaptive_eps] = 0.0

    # 1. 剖分与重网格化（放宽筛选阈值至 2.5 * typical_edge 以防漏切）
    for f_idx, face in enumerate(faces):
        dist = dists[f_idx]
        s = s_all[face]
        is_near = dist < 2.5 * typical_edge

        if not is_near:
            new_faces.append(face.tolist())
            for idx_local in range(3):
                u, v = face[idx_local], face[(idx_local + 1) % 3]
                if s[idx_local] == 0.0 and s[(idx_local + 1) % 3] == 0.0:
                    plane_edges.add((min(u, v), max(u, v)))
            continue

        zero_mask = s == 0.0
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
            if s_other[0] * s_other[1] < -adaptive_eps:
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

    # 2. 尝试基于平面重剖分边追踪路径
    seam_path = []
    try:
        seam_path = trace_straight_seam(
            plane_edges,
            new_bottom_vertices,
            new_top_vertices,
            new_vertices_arr,
            selected[0],
            selected[-1],
        )
    except Exception as e:
        warnings.warn(f"Direct seam tracing encountered: {e}")

    # 3. 兜底回退保护机制：若平面相交边图断开，回退至全局网格图最短路径搜索
    if len(seam_path) < 2:
        warnings.warn(
            "Plane remeshing failed to form a connected seam path. "
            "Falling back to original mesh surface graph shortest path."
        )
        mesh_adj = {}
        for u, v in mesh_edges_from_faces(faces):
            mesh_adj.setdefault(u, []).append(v)
            mesh_adj.setdefault(v, []).append(u)

        start_v = int(
            bottom_ids[
                np.argmin(np.linalg.norm(vertices[bottom_ids] - selected[0], axis=1))
            ]
        )
        end_v = int(
            top_ids[np.argmin(np.linalg.norm(vertices[top_ids] - selected[-1], axis=1))]
        )

        seam_path = dijkstra_path(mesh_adj, start_v, end_v, vertices)
        if len(seam_path) < 2:
            raise ValueError(
                "Failed to find a valid connected path between boundaries on the mesh."
            )

        remeshed_mesh = mesh
    else:
        remeshed_mesh = trimesh.Trimesh(
            vertices=new_vertices_arr,
            faces=np.array(new_faces, dtype=np.int64),
            process=False,
        )

    # 4. 执行实际切缝
    original_vertex_count = len(remeshed_mesh.vertices)
    cut_mesh, _ = _cut_along_path(remeshed_mesh, np.array(seam_path, dtype=np.int64))

    seam_data = PrecutSeam(
        original_vertex_ids=np.asarray(seam_path, dtype=np.int64),
        cut_vertex_pairs=np.column_stack(
            [
                seam_path,
                np.arange(
                    original_vertex_count,
                    original_vertex_count + len(seam_path),
                    dtype=np.int64,
                ),
            ]
        ),
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
    return max(candidates, key=lambda item: item.stat().st_mtime)


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
    for index in range(len(pairs) - 1):
        a0, b0 = map(int, pairs[index])
        a1, b1 = map(int, pairs[index + 1])
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
    else:
        raise ValueError(f"Unknown trajectory region: {region}")
    if not candidates:
        return None
    return min(item[0] for item in candidates), max(item[1] for item in candidates)


def optimize_spacing(nominal: float, dimensions: np.ndarray) -> Optimum:
    nominal = float(nominal)
    dimensions = np.asarray(dimensions, dtype=float)
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

    counts = np.zeros((len(candidates), 3), dtype=int)
    counts[:, 0] = center_count
    active = dimensions[None, :] > 0.75 * candidates[:, None]
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
    return TrajectorySegment(
        kind="OVERTRAVEL",
        uv=np.full((len(xyz), 2), np.nan),
        xyz=xyz,
        spray_on=False,
        note="3D-only opening connector",
        region="TRANSITION"
    )


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
    return TrajectorySegment(
        "SURFACE_SCAN", uv,
        mapper.map(uv, snap_tolerance, bridge_outside=True), True, note,
        region
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
    if len(xyz) < 3:
        return xyz

    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    keep_mask = np.r_[True, dists > min_distance]
    cleaned_xyz = xyz[keep_mask]

    if len(cleaned_xyz) < 3:
        return cleaned_xyz

    smoothed_xyz = cleaned_xyz.copy()
    for _ in range(smooth_iterations):
        laplacian = 0.5 * (smoothed_xyz[:-2] + smoothed_xyz[2:]) - smoothed_xyz[1:-1]
        smoothed_xyz[1:-1] += smooth_weight * laplacian

    return smoothed_xyz


import numpy as np
import trimesh


def prune_v_sharp_apex(
        xyz: np.ndarray,
        cutoff_distance: float,
        mesh: trimesh.Trimesh | None = None
) -> np.ndarray:
    """
    鲁棒的 V 字狭窄折返区域截断滤波器：
    通过全段两分支交叉距离矩阵，精确截断间距小于 cutoff_distance 的夹角死区。
    """
    if len(xyz) < 8:
        return xyz

    # 1. 拆分为前半段 (进) 与后半段 (出)
    # 对于拼接轨迹，中间索引处即为切缝折返接缝点
    mid_idx = len(xyz) // 2
    leg1 = xyz[:mid_idx]  # 进气支 (0 -> mid)
    leg2 = xyz[mid_idx:]  # 出气支 (mid -> end)

    # 2. 计算两支点对之间的欧氏距离矩阵 (M x N)
    # diffs[i, j] 表示 leg1[i] 到 leg2[j] 的空间距离
    diffs = np.linalg.norm(leg1[:, None, :] - leg2[None, :, :], axis=2)

    # 3. 寻找间距小于 cutoff_distance 的重叠区域
    under_cutoff = diffs < cutoff_distance

    if not np.any(under_cutoff):
        # 两支全程间距均大于截断距离，无需截断
        return xyz

    # 找到两支各自在重叠区之外的最深有效点：
    # leg1 从头向前找最后一个不小于 cutoff 的点
    # leg2 从尾向后找第一个不小于 cutoff 的点
    # 获取所有小于 cutoff 的点对索引
    i_indices, j_indices = np.where(under_cutoff)

    # 截断点：leg1 取进入狭窄区前的点，leg2 取离开狭窄区后的点
    idx1 = max(0, int(np.min(i_indices)) - 1)
    idx2 = min(len(leg2) - 1, int(np.max(j_indices)) + 1)

    if idx1 >= len(leg1) or idx2 <= 0:
        return xyz

    # 4. 取截断点并生成中间桥接过渡点
    p1 = leg1[idx1]
    p2 = leg2[idx2]
    p_mid = 0.5 * (p1 + p2)

    # 5. 拼接新轨迹，彻底剔除尖端重叠尾迹
    new_xyz = np.vstack([
        leg1[:idx1 + 1],
        p_mid[None, :],
        leg2[idx2:]
    ])

    # 6. 贴回曲面
    if mesh is not None:
        proj_mid, _, _ = trimesh.proximity.closest_point(mesh, p_mid[None, :])
        new_xyz[idx1 + 1] = proj_mid[0]

    return new_xyz


# =========================================================================
#                 核心修改：纯 3D 空间样条轨迹后处理优化器
# =========================================================================
def optimize_trajectory_arcs_postprocess(
    trajectory: List[TrajectorySegment],
    mesh: trimesh.Trimesh,
    sample_step: float = 1.0,
    smooth_factor: float = 0.15,
) -> List[TrajectorySegment]:
    optimized: List[TrajectorySegment] = []

    for seg in trajectory:
        if seg.kind == "SURFACE_SCAN" and getattr(seg, "region", "") in ("LOWER", "UPPER"):
            xyz = np.asarray(seg.xyz, dtype=float)
            if len(xyz) < 4:
                optimized.append(seg)
                continue

            # =========================================================
            # ① 核心新增：V 字狭窄区域截断滤波 (以 d/2 截断，中点作为新顶点)
            # =========================================================
            # 这里的 cutoff 设为 0.5 * target_spacing（根据 sample_step 比例估算或直接传值）
            # 截断阈值建议直接设为 0.5 ~ 0.6 倍的实际目标行距 (例如 15mm ~ 18mm)
            cutoff_d = 0.55 * (4.0 * sample_step)  # 如果 sample_step = d/4，此值即 0.55 * d
            xyz_pruned = prune_v_sharp_apex(xyz, cutoff_distance=cutoff_d, mesh=mesh)

            p_start = xyz_pruned[0].copy()
            p_end = xyz_pruned[-1].copy()

            # ② 消除微小重叠点
            dists = np.linalg.norm(np.diff(xyz_pruned, axis=0), axis=1)
            keep_indices = np.concatenate(([True], dists > 1e-4))
            pts = xyz_pruned[keep_indices]

            if len(pts) < 4:
                optimized.append(seg)
                continue

            cumulative_dist = np.insert(np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1)), 0, 0.0)
            total_len = cumulative_dist[-1]

            if total_len <= sample_step:
                optimized.append(seg)
                continue

            try:
                # ③ 3D 参数样条平滑
                weights = np.ones(len(pts))
                weights[0] = 500.0
                weights[-1] = 500.0

                s_val = smooth_factor * len(pts)
                tck, _ = splprep(
                    [pts[:, 0], pts[:, 1], pts[:, 2]],
                    s=s_val,
                    k=min(3, len(pts) - 1),
                    w=weights
                )

                # ④ 等距重采样与曲面投影
                count = max(int(np.ceil(total_len / sample_step)) + 1, 8)
                u_new = np.linspace(0.0, 1.0, count)
                spline_xyz = np.column_stack(splev(u_new, tck))

                proj_xyz, _, _ = trimesh.proximity.closest_point(mesh, spline_xyz)
                proj_xyz[0] = p_start
                proj_xyz[-1] = p_end

                optimized.append(TrajectorySegment(
                    kind=seg.kind,
                    uv=np.full((len(proj_xyz), 2), np.nan),
                    xyz=proj_xyz,
                    spray_on=seg.spray_on,
                    note=seg.note + " [V-Pruned & 3D Spline]",
                    region=seg.region,
                ))
            except Exception:
                optimized.append(seg)
        else:
            optimized.append(seg)

    return optimized


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

    if abs(extents["lower_left"] - extents["lower_right"]) <= 0.1:
        lower_ref_width = extents["lower_left"]
    else:
        lower_ref_width = min(extents["lower_left"], extents["lower_right"])

    if abs(extents["upper_left"] - extents["upper_right"]) <= 0.1:
        upper_ref_width = extents["upper_left"]
    else:
        upper_ref_width = min(extents["upper_left"], extents["upper_right"])

    dimensions = np.asarray([
        frame.waist_width,
        upper_ref_width,
        lower_ref_width,
    ], dtype=float)

    optimum = optimize_spacing(nominal_spacing, dimensions)
    target_spacing = optimum.spacing

    sample_step = target_spacing / 4.0 if sample_step is None else float(sample_step)
    if sample_step <= 0.0:
        raise ValueError("Trajectory sampling step must be greater than zero.")
    boundary_snap_tolerance = max(100.0 * (1.0e-9 * uv_scale), 0.5 * sample_step)

    center_count = int(optimum.counts[0])
    upper_ref_count = int(optimum.counts[1])
    lower_ref_count = int(optimum.counts[2])

    lower_left_count = lower_ref_count
    lower_right_count = lower_ref_count
    lower_left_x, lower_spacing = _balanced_outer_positions(
        lower_ref_width, lower_left_count, 0.0, -1.0
    )
    lower_right_x, _ = _balanced_outer_positions(
        lower_ref_width, lower_right_count, frame.waist_width, 1.0
    )

    upper_left_count = upper_ref_count
    upper_right_count = upper_ref_count
    upper_left_x, upper_spacing = _balanced_outer_positions(
        upper_ref_width, upper_left_count, 0.0, -1.0
    )
    upper_right_x, _ = _balanced_outer_positions(
        upper_ref_width, upper_right_count, frame.waist_width, 1.0
    )

    center_x, center_spacing = _balanced_center_positions(
        frame.waist_width, center_count
    )

    lower_left = _available_tracks(lower_left_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    lower_right = _available_tracks(lower_right_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    upper_left = _available_tracks(upper_left_x, "upper", local_uv, data.uv_faces, tolerance)
    upper_right = _available_tracks(upper_right_x, "upper", local_uv, data.uv_faces, tolerance)

    (
        center_inner,
        rejected_center_boundary_tracks,
        rejected_center_discontinuous_tracks,
    ) = _available_center_tracks(
        center_x, local_uv, data.uv_faces, tolerance
    )
    if not len(center_inner):
        raise ValueError("The waist is too narrow to form an interior central scan.")

    segments: list[TrajectorySegment] = []
    last_surface: TrajectorySegment | None = None
    lower_accepted_x: list[float] = []
    upper_accepted_x: list[float] = []

    def append_surface(
            segment: TrajectorySegment,
            connect_from_previous: bool,
            region_transition: bool = False,
    ) -> None:
        nonlocal last_surface
        if connect_from_previous and last_surface is not None:
            segments.append(_overtravel_segment(
                last_surface, segment, overtravel_factor * target_spacing
            ))
        segments.append(segment)
        last_surface = segment

    # 下半区生成与拼接
    paired_lower = min(len(lower_left), len(lower_right))
    for index in range(paired_lower):
        left_track_data = lower_left[index]
        right_track_data = lower_right[index]

        left_probe = _make_surface_scan(*left_track_data, True, sample_step, frame, mapper)
        right_probe = _make_surface_scan(*right_track_data, True, sample_step, frame, mapper)

        if last_surface is not None:
            left_gap = float(np.linalg.norm(last_surface.xyz[-1] - left_probe.xyz[0]))
            right_gap = float(np.linalg.norm(last_surface.xyz[-1] - right_probe.xyz[0]))
            first_is_left = (left_gap <= right_gap)
        else:
            first_is_left = True

        if first_is_left:
            master_data, slave_data = left_track_data, right_track_data
            labels = ("lower-left upward (Master)", "lower-right downward (Slave)")
            from_a = True
        else:
            master_data, slave_data = right_track_data, left_track_data
            labels = ("lower-right upward (Master)", "lower-left downward (Slave)")
            from_a = False

        first = _make_surface_scan(*master_data, True, sample_step, frame, mapper, note=labels[0])
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=from_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(first.uv, bridge_outside=True)

        second = _make_surface_scan(*slave_data, False, sample_step, frame, mapper, note=labels[1])
        second.uv[0] = destination
        second.xyz = mapper.map(second.uv, bridge_outside=True)

        merged_uv = np.vstack([first.uv, second.uv])
        merged_xyz = np.vstack([first.xyz, second.xyz])
        clean_xyz = smooth_v_trajectory(
            merged_xyz, min_distance=0.2, smooth_iterations=2
        )

        merged_segment = TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=merged_uv,
            xyz=clean_xyz,
            spray_on=True,
            note=f"{labels[0]} -> {labels[1]}",
            region="LOWER"
        )

        append_surface(
            merged_segment, connect_from_previous=(last_surface is not None)
        )
        lower_accepted_x.extend([float(master_data[0]), float(slave_data[0])])

    # 中部腰部区域
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
            region="CENTER"
        )
        append_surface(
            scan,
            connect_from_previous=(last_surface is not None),
            region_transition=(index == 0 and last_surface is not None),
        )

    # 上半区生成与拼接
    paired_upper = min(len(upper_left), len(upper_right))
    for index in range(paired_upper):
        left_track_data = upper_left[index]
        right_track_data = upper_right[index]

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

        first = _make_surface_scan(*master_data, False, sample_step, frame, mapper, note=labels[0])
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=from_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(first.uv, bridge_outside=True)

        second = _make_surface_scan(*slave_data, True, sample_step, frame, mapper, note=labels[1])
        second.uv[0] = destination
        second.xyz = mapper.map(second.uv, bridge_outside=True)

        merged_uv = np.vstack([first.uv, second.uv])
        merged_xyz = np.vstack([first.xyz, second.xyz])

        merged_segment = TrajectorySegment(
            kind="SURFACE_SCAN",
            uv=merged_uv,
            xyz=merged_xyz,
            spray_on=True,
            note=f"{labels[0]} -> {labels[1]}",
            region="UPPER"
        )

        append_surface(
            merged_segment,
            connect_from_previous=(last_surface is not None),
            region_transition=(not upper_accepted_x and last_surface is not None),
        )
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
        "overtravel": overtravel_factor * target_spacing,
        "waist_width": frame.waist_width,
        "actual_spacings": {
            "lower_left": lower_spacing,
            "lower_right": lower_spacing,
            "center": center_spacing,
            "upper_left": upper_spacing,
            "upper_right": upper_spacing,
        },
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

    quat = np.asarray([qw, qx, qy, qz], dtype=float)
    return quat / max(float(np.linalg.norm(quat)), EPS)


def _orientation_quaternion(tangent: np.ndarray, inward_normal: np.ndarray) -> np.ndarray:
    y_axis = np.asarray(tangent, dtype=float)
    y_norm = np.linalg.norm(y_axis)
    if y_norm > EPS:
        y_axis /= y_norm
    else:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    z_axis = np.asarray(inward_normal, dtype=float)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > EPS:
        z_axis /= z_norm
    else:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= EPS:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(helper, z_axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = np.cross(helper, z_axis)
        x_norm = np.linalg.norm(x_axis)

    x_axis /= max(x_norm, EPS)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), EPS)

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
    export_points = closest + normals * float(spray_distance)

    segment_lengths = [len(seg.xyz) for seg in segments if len(seg.xyz) > 0]
    split_indices = np.cumsum(segment_lengths)[:-1]
    segment_export_points = np.split(export_points, split_indices)

    segment_tangents = []
    for pts in segment_export_points:
        n_pts = len(pts)
        if n_pts == 0:
            continue
        t = np.empty_like(pts)
        if n_pts == 1:
            t[0] = np.array([1.0, 0.0, 0.0])
        else:
            t[0] = pts[1] - pts[0]
            t[-1] = pts[-1] - pts[-2]
            t[1:-1] = pts[2:] - pts[:-2]
            dz = pts[-1, 2] - pts[0, 2]
            if dz < 0:
                t = -t
        segment_tangents.append(t)

    tangents = np.vstack(segment_tangents)
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


def _finite_polyline_parts(points_array: np.ndarray) -> list[np.ndarray]:
    points = np.asarray(points_array, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return []
    finite = np.isfinite(points[:, :3]).all(axis=1)
    parts = []
    start = None
    for index, valid in enumerate(finite):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            if index - start >= 2:
                parts.append(points[start:index, :3])
            start = None
    if start is not None and len(points) - start >= 2:
        parts.append(points[start:, :3])
    return parts


def vtk_tube(polydata: vtk.vtkPolyData, radius: float = 0.4, num_sides: int = 8) -> vtk.vtkPolyData:
    # Keep trajectories as native polylines.  TubeFilter expands every
    # segment into polygons and makes camera interaction lag on dense paths.
    return polydata


def vtk_actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float], opacity: float = 0.90,
              width: float = 1.0, wireframe: bool = False) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetResolveCoincidentTopologyToPolygonOffset()
    try:
        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)
        mapper.SetResolveCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
    except AttributeError:
        pass

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetRenderLinesAsTubes(False)

    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
    return actor


def _style_trajectory_actor(actor: vtk.vtkActor, color: tuple[float, float, float]) -> vtk.vtkActor:
    """Make trajectory geometry readable against the shaded mesh."""
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetAmbient(1.0)
    actor.GetProperty().SetDiffuse(0.0)
    if hasattr(actor.GetProperty(), "SetDisplayLocationToForeground"):
        actor.GetProperty().SetDisplayLocationToForeground()
    actor.SetForceOpaque(True)
    return actor


def add_title(renderer: vtk.vtkRenderer, text: str) -> None:
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.SetPosition(18, 18)
    actor.GetTextProperty().SetFontSize(16)
    actor.GetTextProperty().SetColor(0.12, 0.12, 0.12)
    renderer.AddViewProp(actor)


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

        self.hide_model = False
        self.hide_traj = False

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("OptCuts trajectory - VTK/OpenGL (Interactive)")
        self.window.SetSize(1600, 900)
        self.window.SetMultiSamples(0)
        self.window.SetNumberOfLayers(2)

        self.left_renderer = vtk.vtkRenderer()
        self.left_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
        self.left_renderer.SetLayer(0)
        self.left_renderer.SetBackground(0.96, 0.97, 0.98)

        self.left_overlay_renderer = vtk.vtkRenderer()
        self.left_overlay_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
        self.left_overlay_renderer.SetLayer(1)
        # Render the trajectory as a true foreground overlay, matching Aeigen.
        self.left_overlay_renderer.PreserveColorBufferOn()
        self.left_overlay_renderer.PreserveDepthBufferOff()
        self.left_overlay_renderer.EraseOff()
        self.left_overlay_renderer.SetActiveCamera(self.left_renderer.GetActiveCamera())

        self.right_renderer = vtk.vtkRenderer()
        self.right_renderer.SetViewport(0.5, 0.34, 1.0, 1.0)
        self.right_renderer.SetLayer(0)
        self.right_renderer.SetBackground(0.96, 0.97, 0.98)

        self.chart_renderer = vtk.vtkRenderer()
        self.chart_renderer.SetViewport(0.5, 0.0, 1.0, 0.34)
        self.chart_renderer.SetLayer(0)
        self.chart_renderer.SetBackground(0.96, 0.97, 0.98)

        for ren in (self.left_renderer, self.left_overlay_renderer, self.right_renderer, self.chart_renderer):
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

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetDesiredUpdateRate(30.0)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    def apply_visibility(self):
        self.mesh_actor_3d.SetVisibility(not self.hide_model)
        self.mesh_actor_2d.SetVisibility(not self.hide_model)
        for actor in self.trajectory_actors_3d:
            actor.SetVisibility(not self.hide_traj)
        for actor in self.trajectory_actors_2d:
            actor.SetVisibility(not self.hide_traj)
        self.window.Render()

    def update_trajectory(self, trajectory: Sequence[TrajectorySegment], planning_frame: PlanningFrame,
                          metadata: dict[str, object],
                          trajectory_2d: Sequence[TrajectorySegment] | None = None):
        self.current_trajectory = list(trajectory)
        self.current_metadata = metadata
        display_2d = trajectory_2d if trajectory_2d is not None else trajectory

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

        scan_append_3d = vtk.vtkAppendPolyData()
        overtravel_append_3d = vtk.vtkAppendPolyData()

        has_scan = False
        has_overtravel = False

        for seg in trajectory:
            if len(seg.xyz) < 2:
                continue
            for part in _finite_polyline_parts(seg.xyz):
                poly = vtk_polyline(part)
                if seg.kind == "SURFACE_SCAN":
                    scan_append_3d.AddInputData(poly)
                    has_scan = True
                elif seg.kind in ("OVERTRAVEL", "SEAM_BLEND_3D"):
                    overtravel_append_3d.AddInputData(poly)
                    has_overtravel = True

        if has_scan:
            scan_append_3d.Update()
            actor = _style_trajectory_actor(vtk_actor(scan_append_3d.GetOutput(), (1.0, 0.10, 0.02), opacity=1.0, width=4.5), (1.0, 0.10, 0.02))
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

        if has_overtravel:
            overtravel_append_3d.Update()
            actor = _style_trajectory_actor(vtk_actor(overtravel_append_3d.GetOutput(), (0.05, 1.0, 0.08), opacity=1.0, width=4.0), (0.05, 1.0, 0.08))
            self.left_overlay_renderer.AddActor(actor)
            self.trajectory_actors_3d.append(actor)

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
            split_indices = np.where(dists > 5.0 * self.sample_step)[0] + 1
            for line in np.split(uv_arr, split_indices):
                if len(line) >= 2:
                    scan_append_2d.AddInputData(vtk_polyline(line))
                    has_scan_2d = True

        if has_scan_2d:
            scan_append_2d.Update()
            actor_2d = _style_trajectory_actor(vtk_actor(scan_append_2d.GetOutput(), (1.0, 0.05, 0.0), width=5.0), (1.0, 0.05, 0.0))
            self.right_renderer.AddActor(actor_2d)
            self.trajectory_actors_2d.append(actor_2d)

        ref_line_pts = np.vstack([
            planning_frame.origin,
            planning_frame.origin + planning_frame.waist_width * planning_frame.x_axis
        ])
        ref_actor = vtk_actor(vtk_polyline(ref_line_pts), (0.0, 0.8, 0.8), width=3.0)
        self.right_renderer.AddActor(ref_actor)
        self.trajectory_actors_2d.append(ref_actor)

        opt_curve = metadata["optimization_curve"]
        self.chart_context_actor = add_penalty_chart(
            self.chart_renderer,
            np.asarray(opt_curve["spacing"]),
            np.asarray(opt_curve["objective"]),
            metadata["optimized_spacing"],
            metadata["optimization_objective"]
        )

        self.text_actor_2d_info = vtk.vtkTextActor()
        actual_spacings = metadata["actual_spacings"]
        info_text = f"Actual spacing  center: {actual_spacings['center']:.4f}  upper: {actual_spacings['upper_left']:.4f}  lower: {actual_spacings['lower_left']:.4f}"
        self.text_actor_2d_info.SetInput(info_text)
        self.text_actor_2d_info.SetPosition(400, 18)
        self.text_actor_2d_info.GetTextProperty().SetFontSize(16)
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
    def __init__(self, visualizer: InteractiveVisualizer):
        self.visualizer = visualizer
        self.root = tk.Tk()
        self.root.title("控制台 - 属性控制")
        self.root.geometry("360x250+50+50")
        self.root.wm_attributes("-topmost", True)

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

        self.btn = tk.Button(self.root, text="更新轨迹", command=self.update_spacing, font=("Arial", 11), bg="#1948CD",
                             fg="white")
        self.btn.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.export_btn = tk.Button(
            self.root, text="导出轨迹", command=self.export_trajectory,
            font=("Arial", 11), bg="#1948CD", fg="white"
        )
        self.export_btn.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        self.hide_model_var = tk.BooleanVar(value=False)
        self.cb_model = tk.Checkbutton(
            self.root, text="隐藏三维网格模型", variable=self.hide_model_var, command=self.toggle_model,
            font=("Arial", 10)
        )
        self.cb_model.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=2)

        self.hide_traj_var = tk.BooleanVar(value=False)
        self.cb_traj = tk.Checkbutton(
            self.root, text="隐藏喷涂加工轨迹", variable=self.hide_traj_var, command=self.toggle_traj,
            font=("Arial", 10)
        )
        self.cb_traj.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=2)

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
        values = self._read_control_values()
        if values is None:
            return
        new_d, speed, spray_distance = values
        self._store_process_values(new_d, speed, spray_distance)

        # 1. 重新生成初代轨迹（保持 UV 域纯净）
        raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
            self.visualizer.data,
            self.visualizer.precut_seam,
            target_spacing=new_d,
            sample_step=self.visualizer.sample_step
        )

        # 2. 纯 3D 后处理空间平滑
        optimized_trajectory = optimize_trajectory_arcs_postprocess(
            raw_trajectory, self.visualizer.mesh, sample_step=self.visualizer.sample_step
        )

        # 3. 3D 显示优化后结果，2D 视图展示初代轨迹
        self.visualizer.update_trajectory(
            optimized_trajectory, planning_frame, metadata, trajectory_2d=raw_trajectory
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
    visualizer = InteractiveVisualizer(
        mesh, data, precut_seam, cut_edges, sample_step, initial_spacing,
        csv_path, metadata_path, paq_path, trajectory_speed, spray_distance
    )
    visualizer.update_trajectory(trajectory, planning_frame, metadata, trajectory_2d=trajectory_2d)
    visualizer.reset_cameras()
    panel = TkControlPanel(visualizer)

    visualizer.interactor.AddObserver("TimerEvent", lambda obj, ev: tick_tkinter(obj, ev, panel))
    visualizer.interactor.CreateRepeatingTimer(10)

    visualizer.window.Render()
    visualizer.interactor.Initialize()
    visualizer.interactor.Start()


def _request_spacing(
        mesh: trimesh.Trimesh,
        data: ObjUVData,
        seam: PrecutSeam,
        cut_edges: np.ndarray,
        initial_spacing: float,
) -> tuple[float | None, PlanningFrame]:
    from vtktrajdisplay import request_spacing

    frame = _planning_frame(_resolve_main_seam_uv_segments(data, seam))
    result = request_spacing(
        mesh, data.uv, data.uv_faces, initial_spacing,
        cut_edges=cut_edges, planning_frame=frame,
    )
    if result is None:
        return None, frame
    if isinstance(result, dict):
        result = result.get("d")
        if result is None:
            raise ValueError("Spacing dialog returned no spacing value.")
    if isinstance(result, (tuple, list, np.ndarray)):
        if not len(result):
            return None, frame
        result = result[0]
    return float(result), frame


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
    selected_path = Path(selected).resolve()
    if selected_path.suffix.lower() == ".json":
        model_path, unfolded_obj = cutopt_cache.load(selected_path)
        mesh = _load_mesh(str(model_path))
        optcuts_mesh, precut_seam = _precut_two_boundaries(mesh)
        if precut_seam is None:
            raise ValueError("Cached source model must have exactly two openings.")
        result_obj = unfolded_obj
        data = _read_obj_uv_data(result_obj)
        optimized_cut_edges = _read_obj_cut_edges(result_obj)
        selected_path = model_path
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
        cutopt_cache.save(selected_path, result_obj)
    if len(precut_seam.edges) and len(optimized_cut_edges):
        cut_edges = np.vstack([precut_seam.edges, optimized_cut_edges])
    elif len(precut_seam.edges):
        cut_edges = precut_seam.edges
    else:
        cut_edges = optimized_cut_edges
    if args.no_show:
        selected_spacing = args.spacing
    else:
        selected_spacing, _ = _request_spacing(
            mesh, data, precut_seam, cut_edges, args.spacing
        )
        if selected_spacing is None:
            print("Trajectory generation cancelled.")
            return
    sample_step_val = selected_spacing / 4.0 if args.sample_step <= 0.0 else args.sample_step

    # 1. 原始规划
    raw_trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
        data, precut_seam, target_spacing=selected_spacing, sample_step=sample_step_val
    )

    # 2. 纯 3D 后处理样条与贴面平滑（不改变 generate_seam_aware_trajectory）
    final_3d_trajectory = optimize_trajectory_arcs_postprocess(
        trajectory=raw_trajectory,
        mesh=mesh,
        sample_step=sample_step_val,
        smooth_factor=0.15,
    )

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

    # 3. 导出平滑后的纯 3D 轨迹
    _export_paq_trajectory(paq_path, mesh, final_3d_trajectory,
                           args.trajectory_speed, args.spray_distance)

    if not args.no_show:
        _show_result(
            mesh, data, precut_seam, cut_edges, final_3d_trajectory, planning_frame, metadata,
            selected_spacing, sample_step_val, csv_path, metadata_path,
            paq_path, args.trajectory_speed, args.spray_distance,
            trajectory_2d=raw_trajectory,
        )
    if not args.no_animation:
        _play_iteration_animation(optcuts_mesh, result_obj)


if __name__ == "__main__":
    main()
