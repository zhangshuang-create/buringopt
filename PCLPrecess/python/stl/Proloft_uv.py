#!/usr/bin/env python3
"""
Build and visualize a UV-like coordinate system on a two-boundary loft surface.

The implementation follows the logic used by BoundaryDetector.cpp:
1. Count mesh edges and keep edges used by exactly one triangle as open borders.
2. Order boundary edges into loops, then classify the upper and lower boundary.
3. Use PCA on the upper boundary and build a vertical reference plane from:
   lower center, upper center, and a point in the upper-boundary PCA second direction.
4. Intersect slicing planes with mesh triangles, stitch intersection segments into curves.
5. Resample each contour by arclength with a consistent v=0 start and direction.

Dependencies:
    pip install trimesh numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import webbrowser
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is strongly recommended, but optional here.
    cKDTree = None


EPS = 1.0e-9

# 倾斜喷涂轨迹参数。角度是轨迹方向相对于物理 UV 平面 U 轴的夹角。
TRAJECTORY_ANGLE_DEG = 60.0
TRAJECTORY_SPACING = 30.0
TRAJECTORY_SAMPLE_STEP = 5.0
TRAJECTORY_VELOCITY = 50.0
TRAJECTORY_PROCESS_VALUE = 100.0


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


@dataclass
class UVBuildResult:
    mesh: trimesh.Trimesh
    uv_grid: np.ndarray
    xyz_grid: np.ndarray
    top_boundary: BoundaryLoop
    bottom_boundary: BoundaryLoop
    pca_center: np.ndarray
    pca_axes: np.ndarray
    pca_values: np.ndarray
    reference_plane: Plane
    reference_curves: List[np.ndarray] = field(default_factory=list)
    start_reference_curve: Optional[np.ndarray] = None
    center_sweep_curve: Optional[np.ndarray] = None
    slice_contours: List[np.ndarray] = field(default_factory=list)
    contour_lengths: Optional[np.ndarray] = None
    uv_valid_mask: Optional[np.ndarray] = None
    overlay_uv_grid: Optional[np.ndarray] = None
    overlay_xyz_grid: Optional[np.ndarray] = None
    overlay_valid_mask: Optional[np.ndarray] = None
    zigzag_uv: Optional[np.ndarray] = None
    zigzag_xyz: Optional[np.ndarray] = None


@dataclass
class ThicknessResult:
    yellow_thickness: np.ndarray
    black_thickness: np.ndarray
    yellow_mean: float
    black_mean: float
    yellow_variance: float
    black_variance: float
    yellow_path_xyz: np.ndarray
    black_path_xyz: np.ndarray
    surface_points: np.ndarray
    surface_uv: np.ndarray


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _as_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Expected an array with shape (N, 3).")
    return arr


def _unit(vec: np.ndarray, name: str = "vector") -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n < EPS:
        raise ValueError(f"Cannot normalize near-zero {name}.")
    return vec / n


def _polygon_normal(points: np.ndarray) -> np.ndarray:
    """Newell polygon normal, same idea as computePolygonNormal in C++."""
    pts = _as_array(points)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    n = np.zeros(3)
    for p0, p1 in zip(pts, np.roll(pts, -1, axis=0)):
        n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    if np.linalg.norm(n) < EPS:
        return np.array([0.0, 0.0, 1.0])
    return n / np.linalg.norm(n)


def _plane_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Plane:
    normal = np.cross(p2 - p1, p3 - p1)
    normal = _unit(normal, "plane normal")
    d = -float(np.dot(normal, p1))
    return Plane(point=np.asarray(p1, dtype=float), normal=normal, d=d)


def _deduplicate_points(points: np.ndarray, tol: float = 1.0e-7) -> np.ndarray:
    pts = _as_array(points)
    if len(pts) == 0:
        return pts
    scale = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1.0)
    qtol = max(tol * scale, tol)
    keys = np.round(pts / qtol).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(keep)]


def _arc_lengths(polyline: np.ndarray, closed: bool = False) -> np.ndarray:
    pts = _as_array(polyline)
    if len(pts) == 0:
        return np.zeros(0)
    work = pts
    if closed and np.linalg.norm(pts[0] - pts[-1]) > EPS:
        work = np.vstack([pts, pts[0]])
    seg_len = np.linalg.norm(np.diff(work, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def _is_closed(polyline: np.ndarray, tol: Optional[float] = None) -> bool:
    if len(polyline) < 3:
        return False
    if tol is None:
        tol = max(1.0e-6 * np.linalg.norm(np.ptp(polyline, axis=0)), 1.0e-8)
    return np.linalg.norm(polyline[0] - polyline[-1]) <= tol


def _close_polyline(polyline: np.ndarray) -> np.ndarray:
    pts = _as_array(polyline)
    if len(pts) >= 3 and not _is_closed(pts):
        return np.vstack([pts, pts[0]])
    return pts


def _cleanup_mesh_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove duplicate and degenerate faces across old/new trimesh APIs."""
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


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _unit(normal, "basis normal")
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, helper), "plane basis u")
    v = _unit(np.cross(n, u), "plane basis v")
    return u, v


def _interp_on_polyline(work: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
        return work[0].copy()
    if s >= arc[-1]:
        return work[-1].copy()
    idx = int(np.searchsorted(arc, s, side="right") - 1)
    idx = min(max(idx, 0), len(work) - 2)
    denom = arc[idx + 1] - arc[idx]
    if denom < EPS:
        return work[idx].copy()
    t = (s - arc[idx]) / denom
    return (1.0 - t) * work[idx] + t * work[idx + 1]


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load STL/OBJ/PLY and return a triangular trimesh.Trimesh."""
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
        raise ValueError(f"Unsupported mesh type loaded from {path}: {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh {path} has no triangles.")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        # Keep the mesh open. This call only removes exact duplicate vertices/faces.
        _cleanup_mesh_faces(mesh)
    return mesh


def extract_boundary_loops(mesh: trimesh.Trimesh) -> List[BoundaryLoop]:
    """Extract ordered open-boundary loops by counting undirected triangle edges."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    edge_count: dict[Tuple[int, int], int] = {}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
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

    starts = sorted(adjacency, key=lambda idx: (len(adjacency[idx]) != 1, idx))
    for start in starts:
        has_unvisited = any(edge_key(start, nb) not in visited_edges for nb in adjacency[start])
        if not has_unvisited:
            continue

        ordered = [start]
        prev = -1
        curr = start
        while True:
            candidates = [
                nb
                for nb in adjacency[curr]
                if nb != prev and edge_key(curr, nb) in edge_set and edge_key(curr, nb) not in visited_edges
            ]
            if not candidates and prev >= 0:
                candidates = [
                    nb
                    for nb in adjacency[curr]
                    if edge_key(curr, nb) in edge_set and edge_key(curr, nb) not in visited_edges
                ]
            if not candidates:
                break
            nxt = candidates[0]
            visited_edges.add(edge_key(curr, nxt))
            if nxt == start:
                break
            ordered.append(nxt)
            prev, curr = curr, nxt
            if len(ordered) > len(boundary_edges) + 2:
                warnings.warn("Boundary tracing stopped because it exceeded expected edge count.")
                break

        pts = vertices[np.asarray(ordered, dtype=np.int64)]
        if len(pts) < 3:
            continue
        center = pts.mean(axis=0)
        normal = _polygon_normal(pts)
        loops.append(
            BoundaryLoop(
                points=pts,
                vertex_indices=np.asarray(ordered, dtype=np.int64),
                center=center,
                normal=normal,
                z_level=float(center[2]),
            )
        )

    if len(loops) < 2:
        raise ValueError(f"Expected at least two boundary loops, found {len(loops)}.")
    loops.sort(key=lambda loop: loop.center[2])
    return loops


def classify_top_bottom_boundaries(boundary_loops: Sequence[BoundaryLoop]) -> Tuple[BoundaryLoop, BoundaryLoop]:
    """Classify bottom/top loops by center height, matching the C++ mid-Z idea."""
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
    """Return center, axes and eigenvalues. axes[0] is the first principal direction."""
    pts = _as_array(points)
    if len(pts) < 3:
        raise ValueError("PCA requires at least three points.")
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    values, axes = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order].T
    # Stabilize signs for repeatability.
    for i in range(axes.shape[0]):
        if axes[i, np.argmax(np.abs(axes[i]))] < 0:
            axes[i] *= -1.0
    return center, axes, values


def _upper_boundary_pca_second_point(top_boundary: BoundaryLoop, pca_second_dir: np.ndarray) -> np.ndarray:
    """Pick an actual-ish upper boundary feature point in the positive second PCA direction."""
    pts = top_boundary.points
    direction = _unit(pca_second_dir, "PCA second direction")
    rel = pts - top_boundary.center
    proj = rel @ direction
    idx = int(np.argmax(proj))
    return pts[idx]


def _upper_boundary_contact_point(top_boundary: BoundaryLoop) -> np.ndarray:
    """
    Match BoundaryDetector.cpp: project the upper boundary into its plane, run a
    small 8-direction search for a large inscribed-circle center, then use the
    nearest polygon edge contact point as p3 for the reference plane.
    """
    pts = _as_array(top_boundary.points)
    if len(pts) < 3:
        raise ValueError("Upper boundary contact point requires at least three points.")

    p2 = top_boundary.center
    normal = _unit(top_boundary.normal, "upper boundary normal")
    u_axis, v_axis = _plane_basis(normal)
    pts2d = np.column_stack(((pts - p2) @ u_axis, (pts - p2) @ v_axis))
    center2d = np.zeros(2)

    def distance_to_polygon(point: np.ndarray) -> Tuple[float, int]:
        min_dist = float("inf")
        min_idx = 0
        for i in range(len(pts2d)):
            j = (i + 1) % len(pts2d)
            a = pts2d[i]
            b = pts2d[j]
            edge = b - a
            denom = float(np.dot(edge, edge))
            if denom < EPS:
                t = 0.0
            else:
                t = float(np.clip(np.dot(point - a, edge) / denom, 0.0, 1.0))
            projection = a + t * edge
            dist = float(np.linalg.norm(point - projection))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return min_dist, min_idx

    current = center2d.copy()
    current_radius, best_contact_index = distance_to_polygon(current)
    scale = max(float(np.linalg.norm(np.ptp(pts2d, axis=0))), 1.0)
    step_size = 0.1 if scale > 10.0 else 0.01 * scale
    for _ in range(100):
        improved = False
        for angle in np.arange(0.0, 2.0 * math.pi, math.pi / 4.0):
            candidate = current + step_size * np.array([math.cos(angle), math.sin(angle)])
            radius, contact_idx = distance_to_polygon(candidate)
            if radius > current_radius:
                current = candidate
                current_radius = radius
                best_contact_index = contact_idx
                improved = True
                break
        if not improved:
            break

    next_idx = (best_contact_index + 1) % len(pts)
    a2d = pts2d[best_contact_index]
    b2d = pts2d[next_idx]
    edge2d = b2d - a2d
    denom = float(np.dot(edge2d, edge2d))
    if denom < EPS:
        t = 0.0
    else:
        t = float(np.clip(np.dot(current - a2d, edge2d) / denom, 0.0, 1.0))
    return pts[best_contact_index] + t * (pts[next_idx] - pts[best_contact_index])


def build_reference_plane(top_center: np.ndarray, bottom_center: np.ndarray, pca_second_point: np.ndarray) -> Plane:
    """Build plane through lower center, upper center and one PCA-second-direction point."""
    return _plane_from_three_points(np.asarray(bottom_center), np.asarray(top_center), np.asarray(pca_second_point))


def _triangle_plane_segment(
    tri_pts: np.ndarray,
    plane: Plane,
    boundary_vertex_mask: Optional[Iterable[bool]] = None,
    eps: float = 1.0e-9,
) -> Optional[np.ndarray]:
    dists = tri_pts @ plane.normal + plane.d
    dists[np.abs(dists) < eps] = 0.0
    points = []
    for i in range(3):
        j = (i + 1) % 3
        di = dists[i]
        dj = dists[j]
        pi = tri_pts[i]
        pj = tri_pts[j]
        if di == 0.0:
            points.append(pi)
        if di * dj < -eps:
            t = abs(di) / (abs(di) + abs(dj))
            points.append(pi + t * (pj - pi))
        elif di == 0.0 and dj == 0.0:
            points.extend([pi, pj])

    if len(points) < 2:
        return None
    pts = _deduplicate_points(np.asarray(points), tol=1.0e-10)
    if len(pts) < 2:
        return None
    if len(pts) > 2:
        # Coplanar triangle edge case: keep the farthest pair as a stable segment.
        max_pair = (0, 1)
        max_dist = -1.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dist = float(np.linalg.norm(pts[a] - pts[b]))
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (a, b)
        pts = pts[list(max_pair)]
    if np.linalg.norm(pts[0] - pts[1]) < eps:
        return None
    return pts


def _segments_to_curves(segments: Sequence[np.ndarray], tol: float) -> List[np.ndarray]:
    if not segments:
        return []

    nodes: List[np.ndarray] = []
    key_to_idx: dict[Tuple[int, int, int], int] = {}
    adjacency: dict[int, List[int]] = {}

    def node_index(pt: np.ndarray) -> int:
        key = tuple(np.round(pt / tol).astype(np.int64).tolist())
        if key not in key_to_idx:
            key_to_idx[key] = len(nodes)
            nodes.append(np.asarray(pt, dtype=float))
        return key_to_idx[key]

    for seg in segments:
        a = node_index(seg[0])
        b = node_index(seg[1])
        if a == b:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited_edges: set[Tuple[int, int]] = set()

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    curves: List[np.ndarray] = []
    starts = sorted(adjacency, key=lambda i: (len(adjacency[i]) != 1, i))
    for start in starts:
        while True:
            unused = [nb for nb in adjacency[start] if edge_key(start, nb) not in visited_edges]
            if not unused:
                break
            ordered = [start]
            prev = -1
            curr = start
            while True:
                choices = [
                    nb
                    for nb in adjacency[curr]
                    if nb != prev and edge_key(curr, nb) not in visited_edges
                ]
                if not choices:
                    choices = [nb for nb in adjacency[curr] if edge_key(curr, nb) not in visited_edges]
                if not choices:
                    break
                nxt = choices[0]
                visited_edges.add(edge_key(curr, nxt))
                if nxt == ordered[0]:
                    ordered.append(nxt)
                    break
                ordered.append(nxt)
                prev, curr = curr, nxt
                if len(ordered) > len(nodes) + len(segments) + 2:
                    warnings.warn("Intersection curve tracing exceeded expected graph size.")
                    break
            if len(ordered) >= 2:
                curve = np.vstack([nodes[i] for i in ordered])
                curves.append(curve)

    curves.sort(key=lambda c: _arc_lengths(c, closed=False)[-1] if len(c) > 1 else 0.0, reverse=True)
    return curves


def intersect_mesh_with_plane(mesh: trimesh.Trimesh, plane: Plane, tol: Optional[float] = None) -> List[np.ndarray]:
    """Intersect all mesh triangles with a plane and stitch segments into polylines."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scale = max(float(np.linalg.norm(mesh.extents)), 1.0)
    eps = max(1.0e-10 * scale, 1.0e-9)
    if tol is None:
        tol = max(1.0e-7 * scale, 1.0e-8)
    segments = []
    for face in faces:
        seg = _triangle_plane_segment(vertices[face], plane, eps=eps)
        if seg is not None:
            segments.append(seg)
    return _segments_to_curves(segments, tol=tol)


def sort_polyline_points(points: np.ndarray) -> np.ndarray:
    """
    Sort unordered points into a contour.

    For curves already produced by intersect_mesh_with_plane this is usually not needed,
    but this function is useful for external/debug point sets. Closed contours are angle
    sorted in their PCA plane. Open noisy curves use nearest-neighbor chaining.
    """
    pts = _deduplicate_points(_as_array(points))
    if len(pts) < 3:
        return pts
    center, axes, _ = compute_pca(pts)
    normal = axes[-1]
    a0, a1 = axes[0], axes[1]
    proj = np.column_stack(((pts - center) @ a0, (pts - center) @ a1))
    radii = np.linalg.norm(proj, axis=1)
    if np.percentile(radii, 75) > EPS:
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        return pts[np.argsort(angles)]

    remaining = set(range(len(pts)))
    start = int(np.argmin(pts[:, 2]))
    order = [start]
    remaining.remove(start)
    while remaining:
        curr = pts[order[-1]]
        nxt = min(remaining, key=lambda i: float(np.linalg.norm(pts[i] - curr)))
        remaining.remove(nxt)
        order.append(nxt)
    return pts[order]


def choose_start_reference_curve(intersection_curves: Sequence[np.ndarray], pca_direction: np.ndarray) -> np.ndarray:
    """Choose the vertical reference curve closer to the positive PCA second direction."""
    if not intersection_curves:
        raise ValueError("No reference-plane intersection curves were found.")
    direction = _unit(pca_direction, "PCA direction")
    scored = []
    for curve in intersection_curves:
        if len(curve) < 2:
            continue
        centroid = curve.mean(axis=0)
        score = float(np.dot(centroid, direction))
        length = float(_arc_lengths(curve)[-1])
        scored.append((score, length, curve))
    if not scored:
        raise ValueError("Reference-plane intersections contain no usable curves.")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def choose_opposite_reference_curve(
    intersection_curves: Sequence[np.ndarray],
    start_reference_curve: np.ndarray,
) -> Optional[np.ndarray]:
    """Choose the reference-plane curve on the opposite side of the start curve."""
    usable = [c for c in intersection_curves if len(c) >= 2 and _arc_lengths(c)[-1] > EPS]
    if len(usable) < 2:
        return None
    start_center = _as_array(start_reference_curve).mean(axis=0)
    candidates = []
    for curve in usable:
        center = curve.mean(axis=0)
        dist = float(np.linalg.norm(center - start_center))
        length = float(_arc_lengths(curve)[-1])
        candidates.append((dist, length, curve))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _resample_open_polyline(polyline: np.ndarray, num_samples: int) -> np.ndarray:
    pts = _as_array(polyline)
    if len(pts) < 2:
        raise ValueError("Open polyline resampling requires at least two points.")
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot resample a zero-length open polyline.")
    samples = np.linspace(0.0, total, num_samples)
    return np.vstack([_interp_on_polyline(pts, arc, s) for s in samples])


def _point_tangent_on_open_polyline(polyline: np.ndarray, u: float) -> Tuple[np.ndarray, np.ndarray]:
    pts = _as_array(polyline)
    if len(pts) < 2:
        raise ValueError("Need at least two points to evaluate a sweep curve.")
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot evaluate a zero-length sweep curve.")
    s = float(np.clip(u, 0.0, 1.0)) * total
    point = _interp_on_polyline(pts, arc, s)
    idx = int(np.searchsorted(arc, s, side="right") - 1)
    idx = min(max(idx, 0), len(pts) - 2)
    tangent = _unit(pts[idx + 1] - pts[idx], "sweep tangent")
    return point, tangent


def build_reference_center_curve(intersection_curves: Sequence[np.ndarray], pca_direction: np.ndarray) -> np.ndarray:
    """
    Match BoundaryDetector::computeLineAttributes behavior for two vertical
    intersection curves: align their directions, resample by arclength, and
    average corresponding points to obtain the middle v=0 reference curve.
    """
    usable = [c for c in intersection_curves if len(c) >= 2 and _arc_lengths(c)[-1] > EPS]
    if len(usable) < 2:
        return choose_start_reference_curve(intersection_curves, pca_direction)

    usable.sort(key=lambda c: _arc_lengths(c)[-1], reverse=True)
    line1 = usable[0]
    line2 = usable[1]
    if np.linalg.norm(line1[0] - line2[0]) > np.linalg.norm(line1[0] - line2[-1]):
        line2 = line2[::-1]

    len1 = float(_arc_lengths(line1)[-1])
    len2 = float(_arc_lengths(line2)[-1])
    max_len = max(len1, len2)
    scale = max(float(np.linalg.norm(np.ptp(np.vstack([line1, line2]), axis=0))), 1.0)
    sample_step = max(0.001, 0.002 * scale)
    sample_n = max(16, int(max_len / sample_step) + 1)
    sample_n = min(sample_n, 1000)

    sample1 = _resample_open_polyline(line1, sample_n)
    sample2 = _resample_open_polyline(line2, sample_n)
    return 0.5 * (sample1 + sample2)


def _slice_plane_from_centers(bottom_center: np.ndarray, top_center: np.ndarray, u: float) -> Plane:
    axis = _unit(top_center - bottom_center, "loft axis")
    point = (1.0 - u) * bottom_center + u * top_center
    d = -float(np.dot(axis, point))
    return Plane(point=point, normal=axis, d=d)


def _choose_slice_curve(curves: Sequence[np.ndarray], expected_center: np.ndarray) -> np.ndarray:
    usable = [c for c in curves if len(c) >= 3]
    if not usable:
        raise ValueError("No usable contour was found on a slicing plane.")
    # Prefer long closed curves near the expected center.
    def score(c: np.ndarray) -> Tuple[int, float, float]:
        length = float(_arc_lengths(c)[-1])
        closed_bonus = 1 if _is_closed(c) else 0
        center_dist = float(np.linalg.norm(c.mean(axis=0) - expected_center))
        return (closed_bonus, length, -center_dist)

    return max(usable, key=score)


def _contour_start_point_by_reference_plane(contour: np.ndarray, reference_plane: Plane, pca_direction: np.ndarray) -> np.ndarray:
    """Find the positive-side intersection of a contour with the vertical reference plane."""
    pts = _close_polyline(contour)
    dists = pts @ reference_plane.normal + reference_plane.d
    candidates = []
    direction = _unit(pca_direction, "PCA direction")
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        d0, d1 = dists[i], dists[i + 1]
        if abs(d0) < EPS:
            candidates.append(p0)
        if d0 * d1 < 0:
            t = abs(d0) / (abs(d0) + abs(d1))
            candidates.append(p0 + t * (p1 - p0))
    if not candidates:
        # For slightly broken contours, fall back to the point with smallest plane distance.
        idx = int(np.argmin(np.abs(dists[:-1])))
        candidates.append(pts[idx])
    center = pts[:-1].mean(axis=0)
    candidates = np.asarray(candidates)
    return candidates[int(np.argmax((candidates - center) @ direction))]


def _project_curve_point_to_contour(
    contour: np.ndarray,
    reference_curve: np.ndarray,
    u: float,
) -> np.ndarray:
    """Project one point of a reference curve to the nearest point on a contour."""
    pts = _close_polyline(contour)
    if len(pts) < 3:
        raise ValueError("Contour is too short to find a reference start point.")

    curve = _as_array(reference_curve)
    if len(curve) == 1:
        target = curve[0]
    else:
        arc = _arc_lengths(curve, closed=False)
        target = _interp_on_polyline(curve, arc, float(np.clip(u, 0.0, 1.0)) * float(arc[-1]))

    best_i = 0
    best_t = 0.0
    best_dist = float("inf")
    base = pts[:-1]
    for i in range(len(base)):
        j = (i + 1) % len(base)
        a, b = base[i], base[j]
        edge = b - a
        denom = float(np.dot(edge, edge))
        if denom < EPS:
            t = 0.0
            projection = a
        else:
            t = float(np.clip(np.dot(target - a, edge) / denom, 0.0, 1.0))
            projection = a + t * edge
        dist = float(np.linalg.norm(projection - target))
        if dist < best_dist:
            best_dist = dist
            best_i = i
            best_t = t

    a = base[best_i]
    b = base[(best_i + 1) % len(base)]
    return (1.0 - best_t) * a + best_t * b


def _rotate_closed_polyline_to_start(polyline: np.ndarray, start_point: np.ndarray) -> np.ndarray:
    pts = _close_polyline(polyline)
    base = pts[:-1] if _is_closed(pts) else pts
    if len(base) < 3:
        return base

    best_i = 0
    best_t = 0.0
    best_dist = float("inf")
    for i in range(len(base)):
        j = (i + 1) % len(base)
        a, b = base[i], base[j]
        edge = b - a
        denom = float(np.dot(edge, edge))
        if denom < EPS:
            t = 0.0
            proj = a
        else:
            t = float(np.clip(np.dot(start_point - a, edge) / denom, 0.0, 1.0))
            proj = a + t * edge
        dist = float(np.linalg.norm(proj - start_point))
        if dist < best_dist:
            best_dist = dist
            best_i = i
            best_t = t

    i = best_i
    j = (i + 1) % len(base)
    start = (1.0 - best_t) * base[i] + best_t * base[j]
    rotated = [start]
    if best_t < 1.0 - 1.0e-8:
        rotated.append(base[j])
        next_idx = (j + 1) % len(base)
    else:
        next_idx = (j + 1) % len(base)
    while next_idx != i:
        rotated.append(base[next_idx])
        next_idx = (next_idx + 1) % len(base)
    rotated.append(base[i])
    rotated.append(start)
    return np.asarray(rotated)


def _ensure_contour_orientation(polyline: np.ndarray, plane_normal: np.ndarray, preferred_sign: float) -> np.ndarray:
    pts = _close_polyline(polyline)
    n = _polygon_normal(pts[:-1])
    sign = float(np.dot(n, plane_normal))
    if sign * preferred_sign < 0:
        rev = pts[::-1]
        return rev
    return pts


def resample_polyline_by_arclength(polyline: np.ndarray, num_v: int, start_point: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample a closed contour from start_point using normalized arclength."""
    samples, _, _ = resample_polyline_by_arclength_with_values(polyline, num_v, start_point=start_point)
    return samples


def resample_polyline_by_arclength_with_values(
    polyline: np.ndarray,
    num_v: int,
    start_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample a closed contour and return points, local arclengths, and contour length."""
    if num_v < 2:
        raise ValueError("num_v must be >= 2.")
    pts = _as_array(polyline)
    if start_point is not None:
        pts = _rotate_closed_polyline_to_start(pts, np.asarray(start_point, dtype=float))
    pts = _close_polyline(pts)
    if len(pts) < 4:
        raise ValueError("A closed contour needs at least three distinct points.")
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot resample a zero-length contour.")
    samples = np.linspace(0.0, total, num_v, endpoint=True)
    return np.vstack([_interp_on_polyline(pts, arc, s) for s in samples]), samples, total


def sample_polyline_by_normalized_values(
    polyline: np.ndarray,
    normalized_values: np.ndarray,
    start_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a closed contour at prescribed normalized arc positions in [0, 1]."""
    pts = _as_array(polyline)
    if start_point is not None:
        pts = _rotate_closed_polyline_to_start(pts, np.asarray(start_point, dtype=float))
    pts = _close_polyline(pts)
    if len(pts) < 4:
        raise ValueError("A closed contour needs at least three distinct points.")
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot sample a zero-length contour.")
    normalized = np.clip(np.asarray(normalized_values, dtype=float), 0.0, 1.0)
    arc_values = normalized * total
    samples = np.vstack([_interp_on_polyline(pts, arc, s) for s in arc_values])
    return samples, arc_values, total


def sample_polyline_by_arclength_values(
    polyline: np.ndarray,
    arc_values: np.ndarray,
    start_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a closed contour at prescribed absolute arclength positions."""
    pts = _as_array(polyline)
    if start_point is not None:
        pts = _rotate_closed_polyline_to_start(pts, np.asarray(start_point, dtype=float))
    pts = _close_polyline(pts)
    if len(pts) < 4:
        raise ValueError("A closed contour needs at least three distinct points.")
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot sample a zero-length contour.")
    values = np.clip(np.asarray(arc_values, dtype=float), 0.0, total)
    samples = np.vstack([_interp_on_polyline(pts, arc, s) for s in values])
    return samples, values, total


def _closed_polyline_arc_at_point(polyline: np.ndarray, point: np.ndarray) -> Tuple[float, float]:
    pts = _close_polyline(polyline)
    if len(pts) < 4:
        raise ValueError("A closed contour needs at least three distinct points.")
    arc = _arc_lengths(pts, closed=False)
    base = pts[:-1]
    best_s = 0.0
    best_dist = float("inf")
    for i in range(len(base)):
        a = pts[i]
        b = pts[i + 1]
        edge = b - a
        denom = float(np.dot(edge, edge))
        if denom < EPS:
            t = 0.0
            projection = a
        else:
            t = float(np.clip(np.dot(point - a, edge) / denom, 0.0, 1.0))
            projection = a + t * edge
        dist = float(np.linalg.norm(projection - point))
        if dist < best_dist:
            best_dist = dist
            best_s = float(arc[i] + t * (arc[i + 1] - arc[i]))
    return best_s, float(arc[-1])


def sample_closed_polyline_two_sided(
    polyline: np.ndarray,
    num_samples: int,
    start_point: np.ndarray,
    opposite_point: np.ndarray,
    u_extent: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Sample a closed contour from u=0 toward the opposite point in both directions."""
    if num_samples < 3:
        raise ValueError("num_samples must be >= 3 for two-sided contour sampling.")
    pts = _close_polyline(polyline)
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot sample a zero-length contour.")

    start_s, _ = _closed_polyline_arc_at_point(pts, start_point)
    opposite_s, _ = _closed_polyline_arc_at_point(pts, opposite_point)
    ccw_len = (opposite_s - start_s) % total
    cw_len = (start_s - opposite_s) % total
    if ccw_len < EPS or cw_len < EPS:
        raise ValueError("Start and opposite points collapsed on the contour.")

    neg_count = num_samples // 2 + 1
    pos_count = num_samples - neg_count + 1
    neg_dist = np.linspace(cw_len, 0.0, neg_count, endpoint=True)
    pos_dist = np.linspace(0.0, ccw_len, pos_count, endpoint=True)
    neg_arc = (start_s - neg_dist) % total
    pos_arc = (start_s + pos_dist) % total
    sample_arc = np.concatenate([neg_arc, pos_arc[1:]])

    if u_extent is None:
        neg_u = -neg_dist
        pos_u = pos_dist
    else:
        neg_u = np.linspace(-u_extent, 0.0, neg_count, endpoint=True)
        pos_u = np.linspace(0.0, u_extent, pos_count, endpoint=True)
    u_values = np.concatenate([neg_u, pos_u[1:]])
    samples = np.vstack([_interp_on_polyline(pts, arc, s) for s in sample_arc])
    return samples, u_values, sample_arc, total, cw_len, ccw_len


def sample_closed_polyline_two_sided_by_u_values(
    polyline: np.ndarray,
    u_values: np.ndarray,
    start_point: np.ndarray,
    opposite_point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample a closed contour using signed arclength from the u=0 point."""
    pts = _close_polyline(polyline)
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        raise ValueError("Cannot sample a zero-length contour.")

    start_s, _ = _closed_polyline_arc_at_point(pts, start_point)
    opposite_s, _ = _closed_polyline_arc_at_point(pts, opposite_point)
    ccw_len = (opposite_s - start_s) % total
    cw_len = (start_s - opposite_s) % total
    if ccw_len < EPS or cw_len < EPS:
        raise ValueError("Start and opposite points collapsed on the contour.")

    u_values = np.asarray(u_values, dtype=float)
    clipped = np.clip(u_values, -cw_len, ccw_len)
    arc_values = (start_s + clipped) % total
    samples = np.vstack([_interp_on_polyline(pts, arc, s) for s in arc_values])
    return samples, arc_values, total


def slice_loft_surface(
    mesh: trimesh.Trimesh,
    num_u: int,
    top_boundary: Optional[BoundaryLoop] = None,
    bottom_boundary: Optional[BoundaryLoop] = None,
    sweep_curve: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Generate contours by sweeping along the center curve, falling back to horizontal planes."""
    if num_u < 2:
        raise ValueError("num_u must be >= 2.")
    if top_boundary is None or bottom_boundary is None:
        loops = extract_boundary_loops(mesh)
        top_boundary, bottom_boundary = classify_top_bottom_boundaries(loops)

    contours: List[np.ndarray] = []
    for i, u in enumerate(np.linspace(0.0, 1.0, num_u)):
        if i == 0:
            contours.append(_close_polyline(bottom_boundary.points))
            continue
        if i == num_u - 1:
            contours.append(_close_polyline(top_boundary.points))
            continue
        if sweep_curve is not None and len(sweep_curve) >= 2:
            point, tangent = _point_tangent_on_open_polyline(sweep_curve, float(u))
            plane = Plane(point=point, normal=tangent, d=-float(np.dot(tangent, point)))
        else:
            plane = _slice_plane_from_centers(bottom_boundary.center, top_boundary.center, float(u))
        curves = intersect_mesh_with_plane(mesh, plane)
        if not curves:
            raise ValueError(f"Slicing plane {i}/{num_u - 1} produced no intersection.")
        expected = (1.0 - u) * bottom_boundary.center + u * top_boundary.center
        if sweep_curve is not None and len(sweep_curve) >= 2:
            expected, _ = _point_tangent_on_open_polyline(sweep_curve, float(u))
        contours.append(_choose_slice_curve(curves, expected))
    return contours


def _align_layer_direction(curr: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
    if prev is None or len(prev) != len(curr):
        return curr
    same = float(np.mean(np.linalg.norm(curr - prev, axis=1)))
    flipped = curr[::-1]
    flipped = np.roll(flipped, 1, axis=0)
    opposite = float(np.mean(np.linalg.norm(flipped - prev, axis=1)))
    if opposite < same:
        return flipped
    return curr


def _build_zigzag_path(
    uv_grid: np.ndarray,
    xyz_grid: np.ndarray,
    valid_mask: np.ndarray,
    break_between_columns: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    uv_path = []
    xyz_path = []

    def append_cell(cell: Tuple[int, int]):
        r, c = cell
        uv_path.append(uv_grid[r, c])
        xyz_path.append(xyz_grid[r, c])

    def append_break():
        if uv_path and np.isfinite(uv_path[-1]).all():
            uv_path.append(np.array([np.nan, np.nan]))
            xyz_path.append(np.array([np.nan, np.nan, np.nan]))

    for j in range(valid_mask.shape[1]):
        rows = np.where(valid_mask[:, j])[0]
        if len(rows) == 0:
            continue
        if j % 2 == 1:
            rows = rows[::-1]
        for i in rows:
            append_cell((int(i), int(j)))
        if break_between_columns:
            append_break()
    if not uv_path:
        return np.empty((0, 2)), np.empty((0, 3))
    return np.vstack(uv_path), np.vstack(xyz_path)


def _split_valid_mask_into_three_regions(valid_mask: np.ndarray, uv_grid: Optional[np.ndarray] = None) -> List[np.ndarray]:
    if uv_grid is not None and np.any(uv_grid[..., 0] < 0.0):
        return _split_signed_valid_mask_into_regions(valid_mask, uv_grid)

    row_max = np.full(valid_mask.shape[0], -1, dtype=int)
    for i in range(valid_mask.shape[0]):
        cols = np.where(valid_mask[i])[0]
        if len(cols) > 0:
            row_max[i] = int(cols[-1])

    usable = row_max[row_max >= 0]
    if len(usable) == 0:
        return [valid_mask.copy()]

    common_max = int(np.min(usable))
    main = np.zeros_like(valid_mask, dtype=bool)
    main[:, : common_max + 1] = valid_mask[:, : common_max + 1]

    extra = valid_mask.copy()
    extra[:, : common_max + 1] = False
    extra_rows = np.where(np.any(extra, axis=1))[0]
    if len(extra_rows) == 0:
        return [main]

    gaps = np.where(np.diff(extra_rows) > 1)[0]
    if len(gaps) > 0:
        lower_rows = extra_rows[: gaps[0] + 1]
        upper_rows = extra_rows[gaps[0] + 1 :]
    else:
        mid = len(extra_rows) // 2
        lower_rows = extra_rows[:mid]
        upper_rows = extra_rows[mid:]

    lower = np.zeros_like(valid_mask, dtype=bool)
    upper = np.zeros_like(valid_mask, dtype=bool)
    lower[lower_rows, :] = extra[lower_rows, :]
    upper[upper_rows, :] = extra[upper_rows, :]
    return [region for region in (main, lower, upper) if np.any(region)]


def _split_signed_valid_mask_into_regions(valid_mask: np.ndarray, uv_grid: np.ndarray) -> List[np.ndarray]:
    u_values = np.nanmean(uv_grid[..., 0], axis=0)
    zero_col = int(np.nanargmin(np.abs(u_values)))
    regions = []
    for start_col, stop_col in ((zero_col, -1), (zero_col, valid_mask.shape[1])):
        if stop_col < start_col:
            sub = valid_mask[:, : start_col + 1]
            sub = sub[:, ::-1]
        else:
            sub = valid_mask[:, start_col:stop_col]
        for region in _split_valid_mask_into_three_regions(sub):
            if not np.any(region):
                continue
            full = np.zeros_like(valid_mask, dtype=bool)
            if stop_col < start_col:
                full[:, : start_col + 1] = region[:, ::-1]
            else:
                full[:, start_col:stop_col] = region
            regions.append(full)
    return regions


def _build_three_region_zigzag_path(uv_grid: np.ndarray, xyz_grid: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uv_parts = []
    xyz_parts = []
    for region_index, region in enumerate(_split_valid_mask_into_three_regions(valid_mask, uv_grid)):
        uv_part, xyz_part = _build_zigzag_path(
            uv_grid,
            xyz_grid,
            region,
            break_between_columns=False,
        )
        if len(uv_part) == 0:
            continue
        if uv_parts:
            uv_parts.append(np.array([[np.nan, np.nan]]))
            xyz_parts.append(np.array([[np.nan, np.nan, np.nan]]))
        uv_parts.append(uv_part)
        xyz_parts.append(xyz_part)
    if not uv_parts:
        return np.empty((0, 2)), np.empty((0, 3))
    return np.vstack(uv_parts), np.vstack(xyz_parts)


def _finite_segments(points: np.ndarray) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return []
    valid = np.isfinite(pts).all(axis=1)
    segments: List[np.ndarray] = []
    start = None
    for idx, ok in enumerate(valid):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(valid) - 1) and start is not None:
            end = idx + 1 if ok and idx == len(valid) - 1 else idx
            if end - start >= 2:
                segments.append(pts[start:end])
            start = None
    return segments


def _resample_open_polyline(polyline: np.ndarray, spacing: float) -> np.ndarray:
    pts = _as_array(polyline)
    if len(pts) < 2:
        return pts.copy()
    spacing = max(float(spacing), EPS)
    arc = _arc_lengths(pts, closed=False)
    total = float(arc[-1])
    if total < EPS:
        return pts[:1].copy()
    count = max(2, int(math.ceil(total / spacing)) + 1)
    targets = np.linspace(0.0, total, count)
    return np.vstack([_interp_on_polyline(pts, arc, float(s)) for s in targets])


def _resample_nan_separated_path(path: np.ndarray, spacing: float) -> np.ndarray:
    parts = [_resample_open_polyline(seg, spacing) for seg in _finite_segments(path)]
    parts = [part for part in parts if len(part) > 0]
    if not parts:
        return np.empty((0, 3))
    return np.vstack(parts)


def _build_yellow_sweep_path(xyz_grid: np.ndarray, valid_mask: np.ndarray, column_step: int = 1) -> np.ndarray:
    parts = []
    for j in range(0, valid_mask.shape[1], max(1, int(column_step))):
        rows = np.where(valid_mask[:, j])[0]
        if len(rows) < 2:
            continue
        part = xyz_grid[rows, j, :]
        if np.isfinite(part).all(axis=1).sum() >= 2:
            parts.append(part)
    if not parts:
        return np.empty((0, 3))
    separated = []
    for part in parts:
        if separated:
            separated.append(np.array([[np.nan, np.nan, np.nan]]))
        separated.append(part)
    return np.vstack(separated)


def _estimate_grid_normals(xyz_grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    normals = np.full_like(xyz_grid, np.nan, dtype=float)
    rows, cols = valid_mask.shape
    for i in range(rows):
        for j in range(cols):
            if not valid_mask[i, j] or not np.isfinite(xyz_grid[i, j]).all():
                continue

            prev_i = next((k for k in range(i - 1, -1, -1) if valid_mask[k, j] and np.isfinite(xyz_grid[k, j]).all()), None)
            next_i = next((k for k in range(i + 1, rows) if valid_mask[k, j] and np.isfinite(xyz_grid[k, j]).all()), None)
            prev_j = next((k for k in range(j - 1, -1, -1) if valid_mask[i, k] and np.isfinite(xyz_grid[i, k]).all()), None)
            next_j = next((k for k in range(j + 1, cols) if valid_mask[i, k] and np.isfinite(xyz_grid[i, k]).all()), None)

            if prev_i is not None and next_i is not None:
                du = xyz_grid[next_i, j] - xyz_grid[prev_i, j]
            elif next_i is not None:
                du = xyz_grid[next_i, j] - xyz_grid[i, j]
            elif prev_i is not None:
                du = xyz_grid[i, j] - xyz_grid[prev_i, j]
            else:
                continue

            if prev_j is not None and next_j is not None:
                dv = xyz_grid[i, next_j] - xyz_grid[i, prev_j]
            elif next_j is not None:
                dv = xyz_grid[i, next_j] - xyz_grid[i, j]
            elif prev_j is not None:
                dv = xyz_grid[i, j] - xyz_grid[i, prev_j]
            else:
                continue

            n = np.cross(du, dv)
            n_norm = np.linalg.norm(n)
            if n_norm > EPS:
                normals[i, j] = n / n_norm
    return normals


def _gaussian_thickness_from_path(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    path_points: np.ndarray,
    spray_distance: float,
    sigma: float,
    peak: float,
    distance_sigma: float,
) -> np.ndarray:
    if len(surface_points) == 0:
        return np.zeros(0)
    path = _as_array(path_points) if len(path_points) else np.empty((0, 3))
    if len(path) == 0:
        return np.zeros(len(surface_points))

    spray_distance = float(spray_distance)
    sigma = max(float(sigma), EPS)
    distance_sigma = max(float(distance_sigma), EPS)
    thickness = np.zeros(len(surface_points), dtype=float)
    distance_factor = 1.0

    for point_index, (p, n) in enumerate(zip(surface_points, surface_normals)):
        if not np.isfinite(p).all() or not np.isfinite(n).all():
            continue
        n = _unit(n, "surface normal")
        rel = path - p
        axial = rel @ n
        radial_vec = rel - axial[:, None] * n[None, :]
        radial2 = np.einsum("ij,ij->i", radial_vec, radial_vec)
        gaussian = np.exp(-radial2 / (2.0 * sigma * sigma))
        surface_factor = np.exp(-(axial * axial) / (2.0 * sigma * sigma))
        thickness[point_index] = float(np.sum(peak * distance_factor * gaussian * surface_factor))

    return thickness


def compute_thickness_distribution(
    result: UVBuildResult,
    spray_distance: float = 30.0,
    sample_step: float = 5.0,
    sigma: float = 12.0,
    peak: float = 1.0,
    distance_sigma: Optional[float] = None,
) -> ThicknessResult:
    """Compute simple normal-incidence Gaussian thickness for yellow sweep and black zigzag paths."""
    uv_grid = result.overlay_uv_grid if result.overlay_uv_grid is not None else result.uv_grid
    xyz_grid = result.overlay_xyz_grid if result.overlay_xyz_grid is not None else result.xyz_grid
    valid_mask = result.overlay_valid_mask if result.overlay_valid_mask is not None else np.isfinite(xyz_grid).all(axis=2)
    distance_sigma = float(distance_sigma) if distance_sigma is not None else max(float(spray_distance) * 0.25, float(sigma) * 0.5, EPS)

    yellow_path = _build_yellow_sweep_path(xyz_grid, valid_mask)
    black_path = result.zigzag_xyz if result.zigzag_xyz is not None else np.empty((0, 3))
    yellow_samples = _resample_nan_separated_path(yellow_path, sample_step)
    black_samples = _resample_nan_separated_path(black_path, sample_step)

    normals_grid = _estimate_grid_normals(xyz_grid, valid_mask)
    finite = valid_mask & np.isfinite(xyz_grid).all(axis=2) & np.isfinite(normals_grid).all(axis=2) & np.isfinite(uv_grid).all(axis=2)
    surface_points = xyz_grid[finite]
    surface_normals = normals_grid[finite]
    surface_uv = uv_grid[finite]

    yellow = _gaussian_thickness_from_path(surface_points, surface_normals, yellow_samples, spray_distance, sigma, peak, distance_sigma)
    black = _gaussian_thickness_from_path(surface_points, surface_normals, black_samples, spray_distance, sigma, peak, distance_sigma)
    return ThicknessResult(
        yellow_thickness=yellow,
        black_thickness=black,
        yellow_mean=float(np.mean(yellow)) if len(yellow) else float("nan"),
        black_mean=float(np.mean(black)) if len(black) else float("nan"),
        yellow_variance=float(np.var(yellow)) if len(yellow) else float("nan"),
        black_variance=float(np.var(black)) if len(black) else float("nan"),
        yellow_path_xyz=yellow_samples,
        black_path_xyz=black_samples,
        surface_points=surface_points,
        surface_uv=surface_uv,
    )


def build_uv_grid(mesh: trimesh.Trimesh, num_u: int, num_v: int) -> Tuple[np.ndarray, np.ndarray, UVBuildResult]:
    """Build the real UV domain and an overlaid uniform grid clipped to that domain."""
    loops = extract_boundary_loops(mesh)
    top_boundary, bottom_boundary = classify_top_bottom_boundaries(loops)

    pca_center, pca_axes, pca_values = compute_pca(top_boundary.points)
    pca_second_dir = pca_axes[1]
    try:
        pca_second_point = _upper_boundary_contact_point(top_boundary)
    except Exception:
        pca_second_point = _upper_boundary_pca_second_point(top_boundary, pca_second_dir)
    reference_plane = build_reference_plane(top_boundary.center, bottom_boundary.center, pca_second_point)

    reference_curves = intersect_mesh_with_plane(mesh, reference_plane)
    start_reference_curve = choose_start_reference_curve(reference_curves, pca_second_dir)
    opposite_reference_curve = choose_opposite_reference_curve(reference_curves, start_reference_curve)
    center_sweep_curve = build_reference_center_curve(reference_curves, pca_second_dir)

    contours = slice_loft_surface(mesh, num_u, top_boundary, bottom_boundary, sweep_curve=center_sweep_curve)
    axis = _unit(top_boundary.center - bottom_boundary.center, "loft axis")
    preferred_sign = 1.0

    xyz_layers = []
    s_layers = []
    left_lengths = []
    right_lengths = []
    v_values = np.linspace(0.0, 1.0, num_u)
    contour_infos = []
    for i, (v_value, contour) in enumerate(zip(v_values, contours)):
        contour = _ensure_contour_orientation(contour, axis, preferred_sign)
        start_point = _project_curve_point_to_contour(
            contour,
            start_reference_curve,
            float(v_value),
        )
        if opposite_reference_curve is not None:
            opposite_point = _project_curve_point_to_contour(
                contour,
                opposite_reference_curve,
                float(v_value),
            )
        else:
            rotated = _rotate_closed_polyline_to_start(contour, start_point)
            arc = _arc_lengths(rotated, closed=False)
            opposite_point = _interp_on_polyline(rotated, arc, 0.5 * float(arc[-1]))
        contour_infos.append((contour, start_point, opposite_point))

    lengths_arr = []
    for i, (contour, start_point, opposite_point) in enumerate(contour_infos):
        sampled, s_values, _, contour_len, left_len, right_len = sample_closed_polyline_two_sided(
            contour,
            num_v,
            start_point,
            opposite_point,
        )
        xyz_layers.append(sampled)
        s_layers.append(s_values)
        lengths_arr.append(contour_len)
        left_lengths.append(left_len)
        right_lengths.append(right_len)

    xyz_grid = np.stack(xyz_layers, axis=0)
    u_grid = np.stack(s_layers, axis=0)
    vv = np.repeat(v_values[:, None], num_v, axis=1)
    uv_grid = np.stack([u_grid, vv], axis=-1)
    lengths_arr = np.asarray(lengths_arr, dtype=float)
    left_lengths_arr = np.asarray(left_lengths, dtype=float)
    right_lengths_arr = np.asarray(right_lengths, dtype=float)

    overlay_u_values = np.linspace(-float(np.max(left_lengths_arr)), float(np.max(right_lengths_arr)), num_v)
    overlay_uv_layers = []
    overlay_xyz_layers = []
    overlay_valid_layers = []
    for i, (v_value, (contour, start_point, opposite_point)) in enumerate(zip(v_values, contour_infos)):
        valid = (overlay_u_values >= -left_lengths_arr[i] - EPS) & (overlay_u_values <= right_lengths_arr[i] + EPS)
        sampled, _, _ = sample_closed_polyline_two_sided_by_u_values(
            contour,
            overlay_u_values,
            start_point,
            opposite_point,
        )
        overlay_uv = np.column_stack([overlay_u_values, np.full(num_v, v_value)])
        sampled[~valid] = np.nan
        overlay_uv[~valid] = np.nan
        overlay_xyz_layers.append(sampled)
        overlay_uv_layers.append(overlay_uv)
        overlay_valid_layers.append(valid)

    overlay_uv_grid = np.stack(overlay_uv_layers, axis=0)
    overlay_xyz_grid = np.stack(overlay_xyz_layers, axis=0)
    overlay_valid_mask = np.stack(overlay_valid_layers, axis=0)
    zigzag_uv, zigzag_xyz = _build_three_region_zigzag_path(overlay_uv_grid, overlay_xyz_grid, overlay_valid_mask)

    result = UVBuildResult(
        mesh=mesh,
        uv_grid=uv_grid,
        xyz_grid=xyz_grid,
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        pca_center=pca_center,
        pca_axes=pca_axes,
        pca_values=pca_values,
        reference_plane=reference_plane,
        reference_curves=reference_curves,
        start_reference_curve=start_reference_curve,
        center_sweep_curve=center_sweep_curve,
        slice_contours=contours,
        contour_lengths=lengths_arr,
        uv_valid_mask=np.ones(uv_grid.shape[:2], dtype=bool),
        overlay_uv_grid=overlay_uv_grid,
        overlay_xyz_grid=overlay_xyz_grid,
        overlay_valid_mask=overlay_valid_mask,
        zigzag_uv=zigzag_uv,
        zigzag_xyz=zigzag_xyz,
    )
    return uv_grid, xyz_grid, result


def _plot_mesh(
    ax,
    mesh: trimesh.Trimesh,
    face_values: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    max_faces: Optional[int] = None,
):
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if max_faces is not None and len(faces) > max_faces:
        step = int(math.ceil(len(faces) / max_faces))
        faces = faces[::step]
        if face_values is not None:
            face_values = face_values[::step]
    tri = verts[faces]
    if face_values is None:
        color = (0.72, 0.75, 0.78, 0.22)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color=color, linewidth=0.1, edgecolor="none")
    else:
        surf = ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            cmap=cmap,
            linewidth=0.05,
            edgecolor="none",
            alpha=0.75,
        )
        surf.set_array(face_values)
        surf.autoscale()
    return tri


def _set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _draw_closed_line(ax, pts: np.ndarray, color: str, linewidth: float = 1.5, label: Optional[str] = None):
    line = _close_polyline(pts)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=linewidth, label=label)


def _plot_valid_2d_line(ax, points: np.ndarray, color: str, linewidth: float = 0.8):
    valid = np.isfinite(points).all(axis=1)
    start = None
    for idx, ok in enumerate(valid):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(valid) - 1) and start is not None:
            end = idx + 1 if ok and idx == len(valid) - 1 else idx
            if end - start >= 2:
                pts = points[start:end]
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth)
            start = None


def _plot_valid_3d_line(ax, points: np.ndarray, color: str, linewidth: float = 0.8):
    valid = np.isfinite(points).all(axis=1)
    start = None
    for idx, ok in enumerate(valid):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(valid) - 1) and start is not None:
            end = idx + 1 if ok and idx == len(valid) - 1 else idx
            if end - start >= 2:
                pts = points[start:end]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth)
            start = None


def _display_indices(n: int, max_count: Optional[int]) -> range:
    if max_count is None or n <= max_count:
        return range(n)
    step = int(math.ceil(n / max_count))
    return range(0, n, step)


def visualize_uv_grid(
    mesh: trimesh.Trimesh,
    xyz_grid: np.ndarray,
    uv_grid: np.ndarray,
    result: Optional[UVBuildResult] = None,
    output_prefix: Optional[str | Path] = None,
    show: bool = True,
    full_vis: bool = False,
):
    """Output 3D grid, UV plane grid, and u/v color-map visualizations."""
    mpl_cache = Path(__file__).resolve().parent / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError(
            "visualize_uv_grid requires a working matplotlib install. "
            "Use --no-show without --fig-prefix to skip visualization."
        ) from exc

    prefix = Path(output_prefix) if output_prefix else None

    fig = plt.figure(figsize=(15, 5))
    max_faces = None if full_vis else 4000
    max_u_lines = None if full_vis else 24
    max_v_lines = None if full_vis else 32
    max_scatter = None if full_vis else 2500

    ax1 = fig.add_subplot(131, projection="3d")
    _plot_mesh(ax1, mesh, max_faces=max_faces)
    for i in _display_indices(xyz_grid.shape[0], max_u_lines):
        _plot_valid_3d_line(ax1, xyz_grid[i], color="#1f77b4", linewidth=0.8)
    for j in _display_indices(xyz_grid.shape[1], max_v_lines):
        _plot_valid_3d_line(ax1, xyz_grid[:, j, :], color="#ff7f0e", linewidth=0.9)
    if result is not None:
        _draw_closed_line(ax1, result.top_boundary.points, "#d62728", 2.2, "top")
        _draw_closed_line(ax1, result.bottom_boundary.points, "#0000cc", 2.2, "bottom")
        if result.start_reference_curve is not None:
            ref = result.start_reference_curve
            ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="#ff00ff", linewidth=3.0, label="u=0 base")
        if result.center_sweep_curve is not None:
            center = result.center_sweep_curve
            ax1.plot(center[:, 0], center[:, 1], center[:, 2], color="#00c8ff", linewidth=2.4, label="v sweep")
        if result.zigzag_xyz is not None and len(result.zigzag_xyz) >= 2:
            _plot_valid_3d_line(ax1, result.zigzag_xyz, color="#111111", linewidth=1.2)
            ax1.plot([], [], [], color="#111111", linewidth=1.2, label="zigzag")
    ax1.set_title("3D UV grid")
    ax1.legend(loc="best")
    _set_axes_equal(ax1)

    ax2 = fig.add_subplot(132)
    for i in range(uv_grid.shape[0]):
        _plot_valid_2d_line(ax2, uv_grid[i], color="#1f77b4", linewidth=0.8)
    for j in range(uv_grid.shape[1]):
        _plot_valid_2d_line(ax2, uv_grid[:, j], color="#ff7f0e", linewidth=0.8)
    if result is not None and result.overlay_uv_grid is not None:
        overlay_uv = result.overlay_uv_grid
        for i in range(overlay_uv.shape[0]):
            _plot_valid_2d_line(ax2, overlay_uv[i], color="#0099a8", linewidth=1.0)
        for j in range(overlay_uv.shape[1]):
            _plot_valid_2d_line(ax2, overlay_uv[:, j], color="#ff4d5a", linewidth=1.0)
    if result is not None and result.zigzag_uv is not None and len(result.zigzag_uv) >= 2:
        _plot_valid_2d_line(ax2, result.zigzag_uv, color="#111111", linewidth=1.0)
    # u is signed arclength while v is a normalized slice coordinate, so equal
    # plot aspect would flatten the real UV domain into a nearly horizontal line.
    ax2.set_aspect("auto")
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.set_title("UV plane")
    uv_for_limits = result.overlay_uv_grid if result is not None and result.overlay_uv_grid is not None else uv_grid
    finite_u = uv_for_limits[..., 0][np.isfinite(uv_for_limits[..., 0])]
    if len(finite_u) > 0:
        pad_u = max(0.02 * float(np.ptp(finite_u)), 1.0e-6)
        ax2.set_xlim(float(np.min(finite_u)) - pad_u, float(np.max(finite_u)) + pad_u)
    ax2.set_ylim(-0.02, 1.02)

    ax3 = fig.add_subplot(133, projection="3d")
    pts = xyz_grid.reshape(-1, 3)
    u_vals = uv_grid[..., 0].reshape(-1)
    valid_scatter = np.isfinite(pts).all(axis=1) & np.isfinite(u_vals)
    pts = pts[valid_scatter]
    u_vals = u_vals[valid_scatter]
    if max_scatter is not None and len(pts) > max_scatter:
        step = int(math.ceil(len(pts) / max_scatter))
        pts = pts[::step]
        u_vals = u_vals[::step]
    scatter = ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=u_vals, s=8, cmap="viridis")
    if result is not None and result.start_reference_curve is not None:
        ref = result.start_reference_curve
        ax3.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="#ff00ff", linewidth=3.0)
    if result is not None and result.center_sweep_curve is not None:
        center = result.center_sweep_curve
        ax3.plot(center[:, 0], center[:, 1], center[:, 2], color="#00c8ff", linewidth=2.4)
    ax3.set_title("u color map")
    _set_axes_equal(ax3)
    fig.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if prefix:
        fig.savefig(prefix.with_name(prefix.name + "_overview.png"), dpi=180)

    fig_v = plt.figure(figsize=(7, 6))
    axv = fig_v.add_subplot(111, projection="3d")
    pts = xyz_grid.reshape(-1, 3)
    v_vals = uv_grid[..., 1].reshape(-1)
    valid_scatter = np.isfinite(pts).all(axis=1) & np.isfinite(v_vals)
    pts = pts[valid_scatter]
    v_vals = v_vals[valid_scatter]
    if max_scatter is not None and len(pts) > max_scatter:
        step = int(math.ceil(len(pts) / max_scatter))
        pts = pts[::step]
        v_vals = v_vals[::step]
    scatter_v = axv.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=v_vals, s=8, cmap="plasma")
    if result is not None:
        _draw_closed_line(axv, result.top_boundary.points, "#d62728", 2.0)
        _draw_closed_line(axv, result.bottom_boundary.points, "#0000cc", 2.0)
        if result.start_reference_curve is not None:
            ref = result.start_reference_curve
            axv.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="#00ffff", linewidth=3.0)
        if result.center_sweep_curve is not None:
            center = result.center_sweep_curve
            axv.plot(center[:, 0], center[:, 1], center[:, 2], color="#ff00ff", linewidth=2.4)
    axv.set_title("v color map")
    _set_axes_equal(axv)
    fig_v.colorbar(scatter_v, ax=axv, fraction=0.046, pad=0.04)
    fig_v.tight_layout()
    if prefix:
        fig_v.savefig(prefix.with_name(prefix.name + "_v_colormap.png"), dpi=180)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_v)


def visualize_thickness_maps(
    thickness: ThicknessResult,
    output_prefix: str | Path,
    show: bool = False,
):
    mpl_cache = Path(__file__).resolve().parent / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("visualize_thickness_maps requires matplotlib.") from exc

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    def draw_one(values: np.ndarray, title: str, suffix: str, path_xyz: np.ndarray, cmap: str):
        fig = plt.figure(figsize=(12, 5))
        ax3 = fig.add_subplot(121, projection="3d")
        scatter = ax3.scatter(
            thickness.surface_points[:, 0],
            thickness.surface_points[:, 1],
            thickness.surface_points[:, 2],
            c=values,
            s=10,
            cmap=cmap,
        )
        if len(path_xyz) >= 2:
            _plot_valid_3d_line(ax3, path_xyz, color="#111111", linewidth=0.8)
        ax3.set_title(f"{title} 3D heat map\nmean={float(np.mean(values)):.6g}, var={float(np.var(values)):.6g}")
        _set_axes_equal(ax3)
        fig.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(122)
        uv_scatter = ax2.scatter(
            thickness.surface_uv[:, 0],
            thickness.surface_uv[:, 1],
            c=values,
            s=12,
            cmap=cmap,
        )
        ax2.set_xlabel("u")
        ax2.set_ylabel("v")
        ax2.set_title(f"{title} UV heat map")
        ax2.set_aspect("auto")
        fig.colorbar(uv_scatter, ax=ax2, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + suffix), dpi=180)
        if show:
            plt.show()
        else:
            plt.close(fig)

    draw_one(thickness.yellow_thickness, "yellow trajectory thickness", "_yellow_thickness.png", thickness.yellow_path_xyz, "inferno")
    draw_one(thickness.black_thickness, "black trajectory thickness", "_black_thickness.png", thickness.black_path_xyz, "magma")


def _values_on_mesh_vertices(mesh: trimesh.Trimesh, thickness: ThicknessResult, values: np.ndarray) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) == 0 or len(thickness.surface_points) == 0 or len(values) == 0:
        return np.zeros(len(verts), dtype=float)
    if cKDTree is not None:
        tree = cKDTree(thickness.surface_points)
        _, idx = tree.query(verts, k=1)
        return values[np.asarray(idx, dtype=int)]
    diff = verts[:, None, :] - thickness.surface_points[None, :, :]
    idx = np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
    return values[idx]


def _path_segments_for_plotly(path_xyz: np.ndarray) -> List[dict]:
    traces = []
    for segment in _finite_segments(path_xyz):
        if len(segment) < 2:
            continue
        traces.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": segment[:, 0].tolist(),
            "y": segment[:, 1].tolist(),
            "z": segment[:, 2].tolist(),
            "line": {"color": "black", "width": 3},
            "name": "trajectory",
            "showlegend": False,
        })
    return traces


def _write_plotly_mesh_html(
    path: str | Path,
    title: str,
    mesh: trimesh.Trimesh,
    vertex_values: np.ndarray,
    path_xyz: np.ndarray,
    mean_value: float,
    variance_value: float,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    value_min = float(np.nanmin(vertex_values)) if len(vertex_values) else 0.0
    value_max = float(np.nanmax(vertex_values)) if len(vertex_values) else 1.0
    if abs(value_max - value_min) < EPS:
        value_max = value_min + 1.0

    mesh_trace = {
        "type": "mesh3d",
        "x": verts[:, 0].tolist(),
        "y": verts[:, 1].tolist(),
        "z": verts[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "intensity": vertex_values.tolist(),
        "colorscale": "Turbo",
        "cmin": value_min,
        "cmax": value_max,
        "showscale": True,
        "colorbar": {"title": "Thickness"},
        "lighting": {"ambient": 0.55, "diffuse": 0.75, "roughness": 0.8, "specular": 0.12},
        "flatshading": False,
        "name": "model thickness",
    }
    traces = [mesh_trace] + _path_segments_for_plotly(path_xyz)
    layout = {
        "title": f"{title}<br>mean={mean_value:.9g}, variance={variance_value:.9g}",
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
            "zaxis": {"title": "Z"},
        },
        "margin": {"l": 0, "r": 0, "t": 70, "b": 0},
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    html, body, #plot {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    .note {{ position: absolute; left: 12px; bottom: 10px; z-index: 10; background: rgba(255,255,255,0.86); padding: 8px 10px; font-family: Arial, sans-serif; font-size: 13px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <div class="note">Rotate: left drag | Pan: right drag | Zoom: wheel. Lower variance means more uniform thickness.</div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def export_interactive_thickness_models(
    mesh: trimesh.Trimesh,
    thickness: ThicknessResult,
    output_prefix: str | Path,
    open_browser: bool = True,
) -> Tuple[Path, Path]:
    prefix = Path(output_prefix)
    yellow_path = prefix.with_name(prefix.name + "_yellow_model.html")
    black_path = prefix.with_name(prefix.name + "_black_model.html")
    yellow_values = _values_on_mesh_vertices(mesh, thickness, thickness.yellow_thickness)
    black_values = _values_on_mesh_vertices(mesh, thickness, thickness.black_thickness)
    _write_plotly_mesh_html(
        yellow_path,
        "Yellow trajectory model thickness heat map",
        mesh,
        yellow_values,
        thickness.yellow_path_xyz,
        thickness.yellow_mean,
        thickness.yellow_variance,
    )
    _write_plotly_mesh_html(
        black_path,
        "Black zigzag trajectory model thickness heat map",
        mesh,
        black_values,
        thickness.black_path_xyz,
        thickness.black_mean,
        thickness.black_variance,
    )
    if open_browser:
        webbrowser.open(yellow_path.resolve().as_uri())
        webbrowser.open(black_path.resolve().as_uri())
    return yellow_path, black_path


def export_thickness_points(path: str | Path, thickness: ThicknessResult):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["u", "v", "x", "y", "z", "yellow_thickness", "black_thickness"],
        )
        writer.writeheader()
        for uv, xyz, yellow, black in zip(
            thickness.surface_uv,
            thickness.surface_points,
            thickness.yellow_thickness,
            thickness.black_thickness,
        ):
            writer.writerow({
                "u": float(uv[0]),
                "v": float(uv[1]),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "yellow_thickness": float(yellow),
                "black_thickness": float(black),
            })


def thickness_summary_text(thickness: ThicknessResult) -> str:
    better = "yellow" if thickness.yellow_variance < thickness.black_variance else "black"
    return (
        "Thickness statistics\n"
        f"Yellow trajectory mean thickness: {thickness.yellow_mean:.9g}\n"
        f"Yellow trajectory thickness variance: {thickness.yellow_variance:.9g}\n"
        f"Black trajectory mean thickness: {thickness.black_mean:.9g}\n"
        f"Black trajectory thickness variance: {thickness.black_variance:.9g}\n"
        f"More uniform by variance: {better}\n\n"
        "Explanation: the mean thickness is the average deposited coating over all valid surface samples. "
        "The variance measures coating uniformity; lower variance means the heat map is more even and has fewer thick/thin bands. "
        "This model uses a simple normal-incidence Gaussian footprint centered on uniformly sampled trajectory points."
    )


def export_thickness_summary(path: str | Path, thickness: ThicknessResult):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(thickness_summary_text(thickness) + "\n", encoding="utf-8")


def export_uv_points(
    path: str | Path,
    uv_grid: np.ndarray,
    xyz_grid: np.ndarray,
    zigzag_uv: Optional[np.ndarray] = None,
    zigzag_xyz: Optional[np.ndarray] = None,
):
    """Save UV/XYZ points as CSV or JSON with fields u, v, x, y, z."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if zigzag_uv is not None and zigzag_xyz is not None and len(zigzag_uv) == len(zigzag_xyz) and len(zigzag_uv) > 0:
        iterable = zip(zigzag_uv, zigzag_xyz)
    else:
        iterable = ((uv_grid[i, j], xyz_grid[i, j]) for i in range(uv_grid.shape[0]) for j in range(uv_grid.shape[1]))

    for uv, xyz in iterable:
        u, v = uv
        x, y, z = xyz
        if not np.all(np.isfinite([u, v, x, y, z])):
            continue
        records.append({"u": float(u), "v": float(v), "x": float(x), "y": float(y), "z": float(z)})

    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["u", "v", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(records)


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion in w, x, y, z order."""
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    n = np.linalg.norm(q)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _nearest_mesh_vertex_normals(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=float)
    if len(vertices) == 0 or len(vertex_normals) != len(vertices):
        return np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))

    if cKDTree is not None:
        _, indices = cKDTree(vertices).query(points)
    else:
        indices = []
        for point in points:
            indices.append(int(np.argmin(np.sum((vertices - point) ** 2, axis=1))))
        indices = np.asarray(indices, dtype=int)

    normals = vertex_normals[indices]
    lengths = np.linalg.norm(normals, axis=1)
    bad = lengths < EPS
    normals[~bad] = normals[~bad] / lengths[~bad, None]
    normals[bad] = np.array([0.0, 0.0, 1.0])
    return normals


def _orientation_quaternion_from_tangent_normal(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    y_axis = _unit(tangent, "trajectory tangent")
    z_axis = _unit(normal, "surface normal")
    x_axis = np.cross(y_axis, z_axis)
    if np.linalg.norm(x_axis) < EPS:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, z_axis))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(helper, z_axis)
    x_axis = _unit(x_axis, "trajectory x axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "trajectory y axis")

    rotation = np.column_stack((-x_axis, y_axis, -z_axis))
    return _matrix_to_quaternion_wxyz(rotation)


def export_black_paq(
    path: str | Path,
    mesh: trimesh.Trimesh,
    black_path_xyz: np.ndarray,
    velocity: float = 50.0,
    acceleration: float = 0.0,
    process_value: float = 100.0,
) -> int:
    """Export the black zigzag trajectory in the C++ new_PAQ.txt row format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    segments = _finite_segments(black_path_xyz)
    if not segments:
        raise ValueError("No valid black trajectory points are available for PAQ export.")

    all_points = np.vstack(segments)
    all_normals = _nearest_mesh_vertex_normals(mesh, all_points)
    normal_offset = 0
    timestamp = 0.0
    rows = []

    for segment in segments:
        normals = all_normals[normal_offset: normal_offset + len(segment)]
        normal_offset += len(segment)
        for i, point in enumerate(segment):
            if i == 0:
                tangent = segment[1] - segment[0]
            elif i == len(segment) - 1:
                tangent = segment[-1] - segment[-2]
                timestamp += float(np.linalg.norm(segment[i] - segment[i - 1])) / max(float(velocity), EPS)
            else:
                tangent = segment[i + 1] - segment[i - 1]
                timestamp += float(np.linalg.norm(segment[i] - segment[i - 1])) / max(float(velocity), EPS)

            if np.linalg.norm(tangent) < EPS:
                tangent = np.array([0.0, 1.0, 0.0])
            quaternion = _orientation_quaternion_from_tangent_normal(tangent, normals[i])
            rows.append((
                float(point[0]), float(point[1]), float(point[2]),
                float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]),
                float(velocity), float(acceleration), float(timestamp), 0, float(process_value),
            ))

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(
                f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f} "
                f"{row[3]:.3f} {row[4]:.3f} {row[5]:.3f} {row[6]:.3f} "
                f"{row[7]:.3f} {row[8]:.3f} {row[9]:.6f} {row[10]} {row[11]:.3f}\n"
            )
    return len(rows)


def export_uv_obj(path: str | Path, xyz_grid: np.ndarray, uv_grid: np.ndarray):
    """Optional OBJ export: grid vertices with vt coordinates and quad faces."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nu, nv = xyz_grid.shape[:2]
    lines = ["# UV loft grid exported by loft_uv_parameterization.py\n"]
    index_map: dict[Tuple[int, int], int] = {}
    for i in range(nu):
        for j in range(nv):
            p = xyz_grid[i, j]
            uv = uv_grid[i, j]
            if not np.all(np.isfinite([p[0], p[1], p[2], uv[0], uv[1]])):
                continue
            index_map[(i, j)] = len(index_map) + 1
            lines.append(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
    for i in range(nu):
        for j in range(nv):
            if (i, j) not in index_map:
                continue
            uv = uv_grid[i, j]
            lines.append(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")

    for i in range(nu - 1):
        for j in range(nv - 1):
            keys = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if any(key not in index_map for key in keys):
                continue
            a, b, c, d = [index_map[key] for key in keys]
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c} {d}/{d}\n")
    path.write_text("".join(lines), encoding="utf-8")


def select_mesh_file_dialog() -> Optional[str]:
    """Open a native file dialog and return the selected mesh path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Cannot open file dialog because tkinter is unavailable.") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select loft surface mesh",
        filetypes=[
            ("Mesh files", "*.stl *.obj *.ply"),
            ("STL files", "*.stl"),
            ("OBJ files", "*.obj"),
            ("PLY files", "*.ply"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def _interpolate_overlay_point(
    u: float,
    s: float,
    u_axis: np.ndarray,
    s_axis: np.ndarray,
    xyz_grid: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """在规则化的物理 UV 网格中双线性插值一个曲面点。"""
    if u < u_axis[0] or u > u_axis[-1] or s < s_axis[0] or s > s_axis[-1]:
        return None
    j = min(int(np.searchsorted(u_axis, u, side="right") - 1), len(u_axis) - 2)
    i = min(int(np.searchsorted(s_axis, s, side="right") - 1), len(s_axis) - 2)
    i, j = max(i, 0), max(j, 0)
    corners_valid = valid_mask[i:i + 2, j:j + 2]
    if corners_valid.shape != (2, 2) or not np.all(corners_valid):
        return None
    du = max(float(u_axis[j + 1] - u_axis[j]), EPS)
    ds = max(float(s_axis[i + 1] - s_axis[i]), EPS)
    tu, ts = (u - u_axis[j]) / du, (s - s_axis[i]) / ds
    p00, p01 = xyz_grid[i, j], xyz_grid[i, j + 1]
    p10, p11 = xyz_grid[i + 1, j], xyz_grid[i + 1, j + 1]
    point = ((1 - ts) * ((1 - tu) * p00 + tu * p01)
             + ts * ((1 - tu) * p10 + tu * p11))
    return point if np.all(np.isfinite(point)) else None


def generate_parallel_surface_trajectory(
    result: UVBuildResult,
    angle_deg: float = TRAJECTORY_ANGLE_DEG,
    spacing: float = TRAJECTORY_SPACING,
    sample_step: float = TRAJECTORY_SAMPLE_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成、裁剪并映射斜向等距平行线；NaN 行分隔不连续轨迹段。"""
    uv_grid = result.overlay_uv_grid
    xyz_grid = result.overlay_xyz_grid
    valid = result.overlay_valid_mask
    if uv_grid is None or xyz_grid is None or valid is None:
        raise ValueError("The physical overlay UV grid was not built.")
    if spacing <= 0 or sample_step <= 0:
        raise ValueError("Trajectory spacing and sample step must be positive.")

    # 边界外的 overlay 单元是 NaN，因此从每列/每行的有效单元恢复公共坐标轴。
    u_axis = np.asarray([np.nanmedian(uv_grid[:, j, 0]) for j in range(uv_grid.shape[1])], dtype=float)
    v_axis = np.asarray([np.nanmedian(uv_grid[i, :, 1]) for i in range(uv_grid.shape[0])], dtype=float)
    # v 原本为 0..1；用中心扫掠曲线长度将其转换为与 u 相同的长度单位。
    center = result.center_sweep_curve
    v_scale = float(_arc_lengths(center)[-1]) if center is not None and len(center) > 1 else float(np.linalg.norm(result.mesh.extents))
    s_axis = v_axis * max(v_scale, EPS)
    finite_u = uv_grid[..., 0][valid]
    u_min, u_max = float(np.min(finite_u)), float(np.max(finite_u))
    s_min, s_max = float(s_axis[0]), float(s_axis[-1])

    theta = math.radians(angle_deg)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-direction[1], direction[0]])
    corners = np.array([[u_min, s_min], [u_min, s_max], [u_max, s_min], [u_max, s_max]])
    c_min, c_max = float(np.min(corners @ normal)), float(np.max(corners @ normal))
    offsets = np.arange(math.floor(c_min / spacing) * spacing,
                        math.ceil(c_max / spacing) * spacing + 0.5 * spacing, spacing)
    t_extent = float(np.hypot(u_max - u_min, s_max - s_min)) + 2.0 * sample_step
    uv_segments: List[np.ndarray] = []
    xyz_segments: List[np.ndarray] = []

    for line_index, offset in enumerate(offsets):
        ts = np.arange(-t_extent, t_extent + 0.5 * sample_step, sample_step)
        points_2d = offset * normal[None, :] + ts[:, None] * direction[None, :]
        current_uv: List[np.ndarray] = []
        current_xyz: List[np.ndarray] = []
        line_parts: List[Tuple[np.ndarray, np.ndarray]] = []
        for u, s in points_2d:
            xyz = _interpolate_overlay_point(float(u), float(s), u_axis, s_axis, xyz_grid, valid)
            if xyz is not None:
                current_uv.append(np.array([u, s / max(v_scale, EPS)]))
                current_xyz.append(xyz)
            elif len(current_xyz) >= 2:
                line_parts.append((np.vstack(current_uv), np.vstack(current_xyz)))
                current_uv, current_xyz = [], []
            else:
                current_uv, current_xyz = [], []
        if len(current_xyz) >= 2:
            line_parts.append((np.vstack(current_uv), np.vstack(current_xyz)))
        if line_index % 2:
            line_parts = [(uv[::-1], xyz[::-1]) for uv, xyz in reversed(line_parts)]
        for uv, xyz in line_parts:
            uv_segments.append(uv)
            xyz_segments.append(xyz)

    if not xyz_segments:
        raise ValueError("No inclined trajectory intersects the valid UV surface.")
    sep2, sep3 = np.full((1, 2), np.nan), np.full((1, 3), np.nan)
    uv_path = np.vstack([item for k, seg in enumerate(uv_segments) for item in ((sep2 if k else np.empty((0, 2))), seg)])
    xyz_path = np.vstack([item for k, seg in enumerate(xyz_segments) for item in ((sep3 if k else np.empty((0, 3))), seg)])
    return uv_path, xyz_path


def export_trajectory_txt(path: str | Path, mesh: trimesh.Trimesh, trajectory: np.ndarray) -> int:
    """输出唯一的轨迹文件；空行明确分隔互不连接的曲面线段。"""
    path = Path(path)
    segments = _finite_segments(trajectory)
    point_count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# x y z qw qx qy qz velocity acceleration timestamp process_flag process_value\n")
        timestamp = 0.0
        for segment_index, segment in enumerate(segments):
            normals = _nearest_mesh_vertex_normals(mesh, segment)
            if segment_index:
                stream.write("\n")
            for i, point in enumerate(segment):
                if i:
                    timestamp += float(np.linalg.norm(point - segment[i - 1])) / max(TRAJECTORY_VELOCITY, EPS)
                tangent = segment[min(i + 1, len(segment) - 1)] - segment[max(i - 1, 0)]
                q = _orientation_quaternion_from_tangent_normal(tangent, normals[i])
                stream.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                    f"{TRAJECTORY_VELOCITY:.3f} 0.000 {timestamp:.6f} 0 {TRAJECTORY_PROCESS_VALUE:.3f}\n"
                )
                point_count += 1
    return point_count


def visualize_inclined_trajectory(result: UVBuildResult, uv_path: np.ndarray, xyz_path: np.ndarray) -> None:
    """仅显示网格、红色三维轨迹及物理 UV 轨迹，不保存任何图片。"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    _plot_mesh(ax3d, result.mesh, max_faces=12000)
    _plot_valid_3d_line(ax3d, xyz_path, color="red", linewidth=1.5)
    ax3d.set_title("Inclined surface trajectory")
    _set_axes_equal(ax3d)
    axuv = fig.add_subplot(122)
    _plot_valid_2d_line(axuv, uv_path, color="red", linewidth=1.1)
    axuv.set_title("Physical UV plane")
    axuv.set_xlabel("u (length)")
    axuv.set_ylabel("v (normalized)")
    axuv.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


def run_pipeline(
    mesh_path: str,
    num_u: int,
    num_v: int,
    out_path: str,
    paq_out: str = "",
    obj_path: str = "",
    fig_prefix: str = "",
    thickness_prefix: str = "",
    thickness_out: str = "",
    spray_distance: float = 30.0,
    spray_sample_step: float = 5.0,
    spray_sigma: float = 12.0,
    spray_peak: float = 1.0,
    no_show: bool = False,
    full_vis: bool = False,
    static_thickness_png: bool = False,
) -> Tuple[UVBuildResult, Optional[ThicknessResult]]:
    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    uv_grid, xyz_grid, result = build_uv_grid(mesh, num_u, num_v)
    export_uv_points(out_path, uv_grid, xyz_grid, result.zigzag_uv, result.zigzag_xyz)
    if paq_out:
        paq_count = export_black_paq(
            paq_out,
            mesh,
            result.zigzag_xyz if result.zigzag_xyz is not None else np.empty((0, 3)),
            velocity=50.0,
            acceleration=0.0,
            process_value=100.0,
        )
        print(f"Saved black trajectory PAQ data ({paq_count} points) to {paq_out}")
    if obj_path:
        export_uv_obj(obj_path, xyz_grid, uv_grid)

    thickness: Optional[ThicknessResult] = None
    if thickness_prefix or thickness_out:
        thickness = compute_thickness_distribution(
            result,
            spray_distance=spray_distance,
            sample_step=spray_sample_step,
            sigma=spray_sigma,
            peak=spray_peak,
        )
        print(thickness_summary_text(thickness))
        if thickness_prefix:
            interactive_paths = export_interactive_thickness_models(
                mesh,
                thickness,
                thickness_prefix,
                open_browser=not no_show,
            )
            if static_thickness_png:
                visualize_thickness_maps(
                    thickness,
                    thickness_prefix,
                    show=False,
                )
            export_thickness_summary(Path(thickness_prefix).with_name(Path(thickness_prefix).name + "_summary.txt"), thickness)
            print(f"Saved interactive model heat maps: {interactive_paths[0]} and {interactive_paths[1]}")
        if thickness_out:
            export_thickness_points(thickness_out, thickness)
            print(f"Saved per-point thickness values to {thickness_out}")

    if no_show and not fig_prefix:
        print("Visualization skipped because --no-show was set and --fig-prefix was not provided.")
    else:
        visualize_uv_grid(
            mesh,
            xyz_grid,
            uv_grid,
            result=result,
            output_prefix=fig_prefix or None,
            show=not no_show,
            full_vis=full_vis,
        )

    valid_count = int(len(result.zigzag_uv)) if result.zigzag_uv is not None else int(np.count_nonzero(np.isfinite(uv_grid[..., 0]) & np.isfinite(xyz_grid[..., 0])))
    print(f"Exported {valid_count} UV points to {out_path}")
    print(f"Top center: {result.top_boundary.center}")
    print(f"Bottom center: {result.bottom_boundary.center}")
    print(f"PCA second direction: {result.pca_axes[1]}")
    return result, thickness


def launch_parameter_ui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        raise RuntimeError("Cannot open parameter UI because tkinter is unavailable.") from exc

    root = tk.Tk()
    root.title("Spray UV Thickness Evaluator")
    root.geometry("760x620")

    vars_ = {
        "mesh": tk.StringVar(),
        "num_u": tk.StringVar(value="25"),
        "num_v": tk.StringVar(value="96"),
        "out": tk.StringVar(value="uv_points.csv"),
        "paq_out": tk.StringVar(value="PAQ.txt"),
        "obj": tk.StringVar(value=""),
        "fig_prefix": tk.StringVar(value="output/uv"),
        "thickness_prefix": tk.StringVar(value="output/thickness"),
        "thickness_out": tk.StringVar(value="output/thickness_points.csv"),
        "spray_distance": tk.StringVar(value="30"),
        "spray_sample_step": tk.StringVar(value="5"),
        "spray_sigma": tk.StringVar(value="12"),
        "spray_peak": tk.StringVar(value="1"),
        "no_show": tk.BooleanVar(value=False),
        "full_vis": tk.BooleanVar(value=False),
        "static_thickness_png": tk.BooleanVar(value=False),
    }

    def browse_mesh():
        path = filedialog.askopenfilename(
            title="Select loft surface mesh",
            filetypes=[
                ("Mesh files", "*.stl *.obj *.ply"),
                ("STL files", "*.stl"),
                ("OBJ files", "*.obj"),
                ("PLY files", "*.ply"),
                ("All files", "*.*"),
            ],
        )
        if path:
            vars_["mesh"].set(path)
            base = Path(path).with_suffix("")
            vars_["out"].set(str(base.with_name(base.name + "_uv_points.csv")))
            vars_["paq_out"].set(str(base.with_name(base.name + "_PAQ.txt")))
            vars_["fig_prefix"].set(str(base.with_name(base.name + "_uv")))
            vars_["thickness_prefix"].set(str(base.with_name(base.name + "_thickness")))
            vars_["thickness_out"].set(str(base.with_name(base.name + "_thickness_points.csv")))

    def browse_save(key: str, title: str, default_ext: str = ".csv"):
        path = filedialog.asksaveasfilename(title=title, defaultextension=default_ext)
        if path:
            vars_[key].set(path)

    def append_log(text: str):
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")
        root.update_idletasks()

    def run_from_ui():
        try:
            mesh_path = vars_["mesh"].get().strip()
            if not mesh_path:
                messagebox.showwarning("Missing mesh", "Please select an STL/OBJ/PLY mesh first.")
                return
            append_log("Running thickness evaluation...")
            _, thickness = run_pipeline(
                mesh_path=mesh_path,
                num_u=int(vars_["num_u"].get()),
                num_v=int(vars_["num_v"].get()),
                out_path=vars_["out"].get().strip() or "uv_points.csv",
                paq_out=vars_["paq_out"].get().strip(),
                obj_path=vars_["obj"].get().strip(),
                fig_prefix=vars_["fig_prefix"].get().strip(),
                thickness_prefix=vars_["thickness_prefix"].get().strip(),
                thickness_out=vars_["thickness_out"].get().strip(),
                spray_distance=float(vars_["spray_distance"].get()),
                spray_sample_step=float(vars_["spray_sample_step"].get()),
                spray_sigma=float(vars_["spray_sigma"].get()),
                spray_peak=float(vars_["spray_peak"].get()),
                no_show=bool(vars_["no_show"].get()),
                full_vis=bool(vars_["full_vis"].get()),
                static_thickness_png=bool(vars_["static_thickness_png"].get()),
            )
            if thickness is not None:
                summary = thickness_summary_text(thickness)
                thickness_prefix = vars_["thickness_prefix"].get().strip()
                if thickness_prefix:
                    prefix = Path(thickness_prefix)
                    summary += (
                        "\n\nHeat map files:\n"
                        f"{prefix.with_name(prefix.name + '_yellow_model.html')}\n"
                        f"{prefix.with_name(prefix.name + '_black_model.html')}\n"
                        f"{prefix.with_name(prefix.name + '_summary.txt')}"
                    )
                append_log(summary)
                messagebox.showinfo("Done", summary)
            else:
                append_log("Done. Thickness calculation was skipped because no thickness output was requested.")
                messagebox.showinfo("Done", "UV outputs generated.")
        except Exception as exc:
            append_log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))

    outer = tk.Frame(root, padx=12, pady=12)
    outer.pack(fill="both", expand=True)

    def add_row(row: int, label: str, key: str, browse=None):
        tk.Label(outer, text=label, anchor="w").grid(row=row, column=0, sticky="ew", pady=3)
        tk.Entry(outer, textvariable=vars_[key]).grid(row=row, column=1, sticky="ew", pady=3)
        if browse:
            tk.Button(outer, text="Browse", command=browse).grid(row=row, column=2, padx=6, pady=3)

    outer.columnconfigure(1, weight=1)
    add_row(0, "Mesh", "mesh", browse_mesh)
    add_row(1, "UV output CSV/JSON", "out", lambda: browse_save("out", "Save UV points", ".csv"))
    add_row(2, "Black PAQ output", "paq_out", lambda: browse_save("paq_out", "Save black PAQ trajectory", ".txt"))
    add_row(3, "OBJ output (optional)", "obj", lambda: browse_save("obj", "Save OBJ", ".obj"))
    add_row(4, "Figure prefix", "fig_prefix", lambda: browse_save("fig_prefix", "Set figure prefix", ".png"))
    add_row(5, "Thickness figure prefix", "thickness_prefix", lambda: browse_save("thickness_prefix", "Set thickness figure prefix", ".png"))
    add_row(6, "Thickness CSV", "thickness_out", lambda: browse_save("thickness_out", "Save thickness CSV", ".csv"))
    add_row(7, "num-u layers", "num_u")
    add_row(8, "num-v samples", "num_v")
    add_row(9, "Spray distance", "spray_distance")
    add_row(10, "Trajectory sample step", "spray_sample_step")
    add_row(11, "Gaussian sigma", "spray_sigma")
    add_row(12, "Gaussian peak", "spray_peak")

    tk.Checkbutton(outer, text="Only save HTML/images, do not open browser/windows", variable=vars_["no_show"]).grid(row=13, column=1, sticky="w", pady=3)
    tk.Checkbutton(outer, text="Also export old static PNG thickness maps", variable=vars_["static_thickness_png"]).grid(row=14, column=1, sticky="w", pady=3)
    tk.Checkbutton(outer, text="Full UV visualization", variable=vars_["full_vis"]).grid(row=15, column=1, sticky="w", pady=3)
    tk.Button(outer, text="Run", command=run_from_ui, height=2).grid(row=16, column=1, sticky="ew", pady=10)

    explanation = (
        "Mean thickness: average coating over valid surface samples. "
        "Variance: coating uniformity; lower variance means more even thickness. "
        "Yellow = v-direction sweep lines, Black = zigzag trajectory."
    )
    tk.Label(outer, text=explanation, wraplength=700, justify="left", fg="#444444").grid(row=17, column=0, columnspan=3, sticky="ew", pady=6)

    log = tk.Text(outer, height=8, state="disabled")
    log.grid(row=18, column=0, columnspan=3, sticky="nsew", pady=6)
    outer.rowconfigure(18, weight=1)
    root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a UV grid on a two-boundary loft surface.")
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ/PLY mesh path. If omitted, a file dialog opens.")
    parser.add_argument("--ui", action="store_true", help="Open a parameter input UI.")
    parser.add_argument("--num-u", type=int, default=25, help="Number of horizontal slice layers.")
    parser.add_argument("--num-v", type=int, default=96, help="Number of arclength samples on each layer.")
    parser.add_argument("--out", default="uv_points.csv", help="CSV or JSON output path for u,v,x,y,z.")
    parser.add_argument("--paq-out", default="", help="Optional PAQ.txt output path for the black zigzag trajectory.")
    parser.add_argument("--obj", default="", help="Optional OBJ grid output path with vt coordinates.")
    parser.add_argument("--fig-prefix", default="", help="Optional prefix for saved debug images.")
    parser.add_argument("--thickness-prefix", default="", help="Optional prefix for yellow/black thickness heat-map images.")
    parser.add_argument("--thickness-out", default="", help="Optional CSV output for per-point yellow/black thickness values.")
    parser.add_argument("--spray-distance", type=float, default=30.0, help="Nozzle standoff distance used for trajectory metadata.")
    parser.add_argument("--spray-sample-step", type=float, default=5.0, help="Uniform sampling step along spray trajectories.")
    parser.add_argument("--spray-sigma", type=float, default=12.0, help="Gaussian spray footprint sigma on the surface.")
    parser.add_argument("--spray-peak", type=float, default=1.0, help="Peak thickness contribution of one sampled spray point.")
    parser.add_argument("--static-thickness-png", action="store_true", help="Also export old static PNG thickness maps.")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening browser or matplotlib windows.")
    parser.add_argument("--full-vis", action="store_true", help="Draw all mesh faces, grid lines, and points. Slower to rotate.")
    return parser.parse_args()


def main():
    # 程序不再显示参数 UI，也不接受批量输出选项；启动后只选择一个网格文件。
    mesh_path = select_mesh_file_dialog()
    if not mesh_path:
        raise SystemExit("No mesh file selected.")
    mesh = load_mesh(mesh_path)
    _, _, result = build_uv_grid(mesh, num_u=80, num_v=240)
    uv_path, xyz_path = generate_parallel_surface_trajectory(result)
    result.zigzag_uv = uv_path
    result.zigzag_xyz = xyz_path
    output_path = Path(mesh_path).with_name(Path(mesh_path).stem + "_trajectory.txt")
    count = export_trajectory_txt(output_path, mesh, xyz_path)
    print(f"Saved {count} inclined trajectory points to {output_path}")
    visualize_inclined_trajectory(result, uv_path, xyz_path)


if __name__ == "__main__":
    main()
