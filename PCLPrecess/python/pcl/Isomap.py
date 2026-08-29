#!/usr/bin/env python3
"""Area-uniformly sample an STL surface and embed it in 2D with Isomap.

Dependencies:
    py -m pip install numpy scipy trimesh matplotlib cupy-cuda11x
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

RUNTIME_CACHE = Path(__file__).resolve().parents[2] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(RUNTIME_CACHE / "cupy"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

try:
    import cupy as cp
except Exception:
    cp = None


def load_stl(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} is not a non-empty triangle mesh")
    loaded = loaded.copy()
    loaded.merge_vertices()
    if hasattr(loaded, "unique_faces") and hasattr(loaded, "update_faces"):
        loaded.update_faces(loaded.unique_faces())
    if hasattr(loaded, "nondegenerate_faces") and hasattr(loaded, "update_faces"):
        loaded.update_faces(loaded.nondegenerate_faces())
    loaded.remove_unreferenced_vertices()
    # STL files often contain inconsistent triangle winding. A coherent winding
    # is required for seam-side selection and meaningful UV flip diagnostics.
    trimesh.repair.fix_winding(loaded)
    return loaded


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def mesh_topology(mesh: trimesh.Trimesh) -> dict[str, int | None]:
    """Return topology diagnostics without assuming the mesh is a manifold."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = len(mesh.vertices)
    all_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    all_edges.sort(axis=1)
    unique_edges, incidence = np.unique(all_edges, axis=0, return_counts=True)
    edge_count = len(unique_edges)
    face_count = len(faces)
    boundary_count = len(ordered_boundary_loops(faces))
    euler = vertex_count - edge_count + face_count
    graph = coo_matrix(
        (
            np.ones(2 * edge_count),
            (
                np.concatenate([unique_edges[:, 0], unique_edges[:, 1]]),
                np.concatenate([unique_edges[:, 1], unique_edges[:, 0]]),
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    component_count, _ = connected_components(graph, directed=False)
    boundary_edges = unique_edges[incidence == 1]
    boundary_degree = np.bincount(boundary_edges.reshape(-1), minlength=vertex_count)
    boundary_branches = int(np.count_nonzero((boundary_degree != 0) & (boundary_degree != 2)))
    nonmanifold_edges = int(np.count_nonzero(incidence > 2))
    genus_numerator = 2 - boundary_count - euler
    valid_for_genus = (
        component_count == 1
        and nonmanifold_edges == 0
        and boundary_branches == 0
        and genus_numerator >= 0
        and genus_numerator % 2 == 0
    )
    genus = genus_numerator // 2 if valid_for_genus else None
    return {
        "vertices": vertex_count,
        "edges": edge_count,
        "faces": face_count,
        "boundaries": boundary_count,
        "euler": euler,
        "components": int(component_count),
        "nonmanifold_edges": nonmanifold_edges,
        "boundary_branches": boundary_branches,
        "genus": genus,
    }


def topology_summary(topology: dict[str, int | None]) -> str:
    return (
        f"V={topology['vertices']}, E={topology['edges']}, F={topology['faces']}, "
        f"boundary={topology['boundaries']}, Euler={topology['euler']}, "
        f"components={topology['components']}, nonmanifold_edges={topology['nonmanifold_edges']}, "
        f"boundary_branches={topology['boundary_branches']}, genus={topology['genus']}"
    )


def ordered_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    unused: set[tuple[int, int]] = set()
    for a, b in boundary_edges:
        ia, ib = int(a), int(b)
        adjacency.setdefault(ia, []).append(ib)
        adjacency.setdefault(ib, []).append(ia)
        unused.add((ia, ib))

    loops: list[np.ndarray] = []
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
                break
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def loft_reference_seam(mesh: trimesh.Trimesh):
    import loft_uv_parameterization as loft

    loops = loft.extract_boundary_loops(mesh)
    top, bottom = loft.classify_top_bottom_boundaries(loops)
    _, axes, _ = loft.compute_pca(top.points)
    direction = axes[:, 1]
    try:
        third_point = loft._upper_boundary_contact_point(top)
    except ValueError:
        third_point = loft._upper_boundary_pca_second_point(top, direction)
    plane = loft.build_reference_plane(top.center, bottom.center, third_point)
    curves = loft.intersect_mesh_with_plane(mesh, plane)
    curve = loft.choose_start_reference_curve(curves, direction)
    return np.asarray(curve, dtype=float), plane


def guided_mesh_path(
    mesh: trimesh.Trimesh, start: int, end: int, guide_curve: np.ndarray
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = mesh_edges(np.asarray(mesh.faces, dtype=np.int64))
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    guide_distance, _ = cKDTree(guide_curve).query(midpoints, k=1)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    scale = max(float(np.median(lengths)), np.finfo(float).eps)
    weights = lengths * (1.0 + 8.0 * guide_distance / scale)
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
        raise ValueError("Could not connect the two loft boundaries with a seam")
    path = [int(end)]
    while path[-1] != start:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("Incomplete predecessor chain while constructing seam")
        path.append(predecessor)
    return np.asarray(path[::-1], dtype=np.int64)


def cut_two_boundary_loft(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = ordered_boundary_loops(faces)
    if len(loops) != 2:
        raise ValueError(f"Expected exactly two boundary loops, found {len(loops)}")

    guide_curve, plane = loft_reference_seam(mesh)
    curve_tree = cKDTree(guide_curve)
    endpoints = []
    for loop in loops:
        distances, _ = curve_tree.query(vertices[loop], k=1)
        endpoints.append(int(loop[int(np.argmin(distances))]))
    seam = guided_mesh_path(mesh, endpoints[0], endpoints[1], guide_curve)
    seam_set = set(int(vertex) for vertex in seam)
    duplicate = {vertex: len(vertices) + i for i, vertex in enumerate(seam)}

    centers = np.mean(vertices[faces], axis=1)
    signed = centers @ plane.normal + float(plane.d)
    cut_faces = faces.copy()
    for face_index, tri in enumerate(cut_faces):
        if signed[face_index] < 0.0:
            for local, vertex in enumerate(tri):
                if int(vertex) in seam_set:
                    tri[local] = duplicate[int(vertex)]

    cut_vertices = np.vstack([vertices, vertices[seam]])
    cut_mesh = trimesh.Trimesh(cut_vertices, cut_faces, process=False)
    cut_mesh.remove_unreferenced_vertices()
    after_count = len(ordered_boundary_loops(np.asarray(cut_mesh.faces, dtype=np.int64)))
    if after_count != 1:
        raise ValueError(
            "The loft seam did not create a disk-like mesh: "
            f"expected 1 boundary, found {after_count}"
        )
    return cut_mesh, vertices[seam]


def sample_surface_uniform_density(
    mesh: trimesh.Trimesh,
    density: float,
    rng: np.random.Generator,
    max_points: int = 0,
    boundary_multiplier: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson-like surface samples plus densely and evenly sampled boundaries."""
    if density <= 0:
        raise ValueError("density must be greater than zero")
    if boundary_multiplier < 0:
        raise ValueError("boundary_multiplier must not be negative")

    areas = np.asarray(mesh.area_faces, dtype=float)
    valid = np.isfinite(areas) & (areas > 0.0)
    face_ids = np.flatnonzero(valid)
    areas = areas[valid]
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise ValueError("The STL has no non-degenerate triangles")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    area_count = max(3, int(round(total_area * density)))
    has_boundary = bool(ordered_boundary_loops(faces))
    if max_points > 0:
        if has_boundary and boundary_multiplier > 1.0:
            area_count = min(area_count, max(3, int(max_points) // 2))
        else:
            area_count = min(area_count, max(3, int(max_points)))

    # Random candidates are thinned with a minimum-distance rule. Independent
    # area sampling alone is statistically uniform but visibly forms clusters.
    candidate_count = max(area_count * 12, 100)
    candidate_faces = rng.choice(face_ids, size=candidate_count, p=areas / total_area)
    triangles = np.asarray(mesh.triangles, dtype=float)[candidate_faces]
    r1 = np.sqrt(rng.random(candidate_count))
    r2 = rng.random(candidate_count)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    candidates = (
        w0[:, None] * triangles[:, 0]
        + w1[:, None] * triangles[:, 1]
        + w2[:, None] * triangles[:, 2]
    )
    radius = 0.72 * np.sqrt(total_area / area_count)
    order = rng.permutation(candidate_count)
    accepted: list[int] = []
    cell_size = radius
    grid: dict[tuple[int, int, int], list[int]] = {}
    origin = candidates.min(axis=0)
    for candidate_index in order:
        point = candidates[candidate_index]
        cell = tuple(np.floor((point - origin) / cell_size).astype(np.int64))
        nearby: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby.extend(grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()))
        if nearby and np.any(
            np.linalg.norm(candidates[nearby] - point, axis=1) < radius
        ):
            continue
        accepted.append(int(candidate_index))
        grid.setdefault(cell, []).append(int(candidate_index))
        if len(accepted) >= area_count:
            break

    points = candidates[accepted]
    sampled_face_ids = candidate_faces[accepted]

    # Sample every open boundary at a much smaller, constant arc-length step.
    # Boundary samples are additional, so increasing the multiplier cannot
    # reduce the interior surface coverage.
    interior_count = len(points)
    if boundary_multiplier > 0:
        edge_faces: dict[tuple[int, int], int] = {}
        edge_counts: dict[tuple[int, int], int] = {}
        for face_id, tri in enumerate(faces):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = (min(int(a), int(b)), max(int(a), int(b)))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                edge_faces.setdefault(edge, face_id)
        boundary_step = radius / boundary_multiplier
        boundary_points: list[np.ndarray] = []
        boundary_faces: list[int] = []
        for edge, count in edge_counts.items():
            if count != 1:
                continue
            a, b = vertices[list(edge)]
            segment_count = max(1, int(np.ceil(np.linalg.norm(b - a) / boundary_step)))
            # Exclude the second endpoint to avoid duplicating shared vertices.
            for t in np.arange(segment_count, dtype=float) / segment_count:
                boundary_points.append((1.0 - t) * a + t * b)
                boundary_faces.append(edge_faces[edge])
        if boundary_points:
            points = np.vstack([points, np.asarray(boundary_points)])
            sampled_face_ids = np.concatenate(
                [sampled_face_ids, np.asarray(boundary_faces, dtype=np.int64)]
            )

    if max_points > 0 and len(points) > max_points:
        # Full Isomap has quadratic memory/time cost, so respect the cap while
        # retaining both surface coverage and the requested boundary spacing.
        total_before_cap = len(points)
        boundary_count = len(points) - interior_count
        if boundary_multiplier <= 1.0:
            # Equal boundary/interior spacing: retain the original proportion so
            # the capped cloud remains approximately area-uniform.
            interior_budget = max(
                3, int(round(int(max_points) * interior_count / total_before_cap))
            )
            interior_budget = min(interior_budget, interior_count, int(max_points))
            boundary_budget = min(boundary_count, int(max_points) - interior_budget)
        else:
            # Dense-boundary flattening modes reserve half the point budget for
            # interior coverage and use the remainder for the boundary.
            interior_budget = min(interior_count, max(3, int(max_points) // 2))
            boundary_budget = min(boundary_count, int(max_points) - interior_budget)
            if boundary_count == 0:
                interior_budget = min(interior_count, int(max_points))
            elif boundary_budget == 0 and max_points > 3:
                boundary_budget = 1
                interior_budget = int(max_points) - 1

        interior_indices = np.arange(interior_count, dtype=np.int64)
        if interior_budget < interior_count:
            interior_indices = np.sort(
                rng.choice(interior_indices, size=interior_budget, replace=False)
            )
        boundary_indices = np.empty(0, dtype=np.int64)
        if boundary_budget > 0:
            # Evenly spaced indices avoid random clusters on the boundary.
            boundary_indices = interior_count + np.floor(
                np.arange(boundary_budget) * boundary_count / boundary_budget
            ).astype(np.int64)
        keep = np.concatenate([interior_indices, boundary_indices])
        points = points[keep]
        sampled_face_ids = sampled_face_ids[keep]
        print(
            f"Sample cap applied: {total_before_cap} requested -> {len(points)} kept "
            f"({len(interior_indices)} interior, {len(boundary_indices)} boundary)."
        )
    return points, sampled_face_ids


def point_cloud_boundary_mask(
    points: np.ndarray,
    neighbors: int = 24,
    angle_threshold_degrees: float = 135.0,
) -> np.ndarray:
    """Detect surface boundary samples using only point-cloud neighborhoods.

    Neighbors are projected to each point's PCA tangent plane. Interior points
    are surrounded in angle, while boundary points have a large empty angular
    sector. No mesh vertices, faces, or STL boundary information are used.
    """
    if len(points) < 4:
        raise ValueError("At least four points are required for boundary detection")
    neighbors = min(max(3, neighbors), len(points) - 1)
    threshold = np.deg2rad(angle_threshold_degrees)
    if not 0.0 < threshold < 2.0 * np.pi:
        raise ValueError("boundary angle threshold must be between 0 and 360 degrees")
    _, neighborhoods = cKDTree(points).query(points, k=neighbors + 1)
    mask = np.zeros(len(points), dtype=bool)
    for point_index, neighbor_indices in enumerate(neighborhoods):
        local = points[neighbor_indices[1:]] - points[point_index]
        covariance = local.T @ local / max(len(local), 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        tangent_coordinates = local @ eigenvectors[:, 1:3]
        lengths = np.linalg.norm(tangent_coordinates, axis=1)
        usable = lengths > np.finfo(float).eps
        if np.count_nonzero(usable) < 3:
            continue
        angles = np.sort(
            np.arctan2(tangent_coordinates[usable, 1], tangent_coordinates[usable, 0])
        )
        circular_gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
        mask[point_index] = float(circular_gaps.max()) >= threshold
    return mask


def estimate_boundary_minimum_gradient_directions(
    points: np.ndarray,
    boundary_mask: np.ndarray,
    neighbors: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate normals and least-height-change tangents at boundary samples."""
    boundary_indices = np.flatnonzero(boundary_mask)
    normals = np.zeros((len(points), 3), dtype=float)
    directions = np.zeros((len(points), 3), dtype=float)
    gradient_norms = np.zeros(len(points), dtype=float)
    minimum_directional_gradients = np.zeros(len(points), dtype=float)
    if len(boundary_indices) == 0:
        return normals, directions, gradient_norms, minimum_directional_gradients

    neighbors = min(max(3, neighbors), len(points) - 1)
    _, neighborhoods = cKDTree(points).query(
        points[boundary_indices], k=neighbors + 1
    )
    vertical = np.array([0.0, 0.0, 1.0])
    for point_index, neighbor_indices in zip(boundary_indices, neighborhoods):
        local = points[neighbor_indices[1:]] - points[point_index]
        covariance = local.T @ local / max(len(local), 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        normal /= max(float(np.linalg.norm(normal)), np.finfo(float).eps)
        direction = np.cross(normal, vertical)
        direction_norm = float(np.linalg.norm(direction))
        surface_gradient = vertical - np.dot(vertical, normal) * normal
        gradient_norms[point_index] = np.linalg.norm(surface_gradient)
        if direction_norm <= 1.0e-10:
            # A horizontal tangent plane has zero height gradient in every
            # direction; choose the dominant local tangent for display.
            direction = eigenvectors[:, 2]
            direction_norm = float(np.linalg.norm(direction))
        direction /= max(direction_norm, np.finfo(float).eps)
        # The absolute directional height gradient is the comparison value.
        # A zero value wins immediately.  For a regular tangent plane the
        # direction perpendicular to the surface gradient is exactly such a
        # direction (up to floating-point error).
        minimum_value = abs(float(np.dot(surface_gradient, direction)))
        if minimum_value <= 1.0e-10:
            minimum_value = 0.0
        # The tied directions +d and -d represent one geometric line. Choose a
        # deterministic sign so only one minimum direction is marked.
        dominant = int(np.argmax(np.abs(direction)))
        if direction[dominant] < 0.0:
            direction = -direction
        normals[point_index] = normal
        directions[point_index] = direction
        minimum_directional_gradients[point_index] = minimum_value
    return normals, directions, gradient_norms, minimum_directional_gradients


def save_boundary_directions(
    path: str | Path,
    points: np.ndarray,
    boundary_mask: np.ndarray,
    normals: np.ndarray,
    directions: np.ndarray,
    gradient_norms: np.ndarray,
    minimum_directional_gradients: np.ndarray,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_index", "x", "y", "z", "normal_x", "normal_y", "normal_z",
            "height_gradient_norm", "min_directional_gradient",
            "min_dir_x", "min_dir_y", "min_dir_z",
        ])
        for index in np.flatnonzero(boundary_mask):
            writer.writerow([
                int(index), *points[index], *normals[index], gradient_norms[index],
                minimum_directional_gradients[index], *directions[index],
            ])


def gpu_is_available() -> tuple[bool, str]:
    if cp is None:
        return False, "CuPy is not installed"
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return False, "CuPy found no CUDA device"
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return True, str(name)
    except Exception as exc:
        return False, f"CUDA initialization failed: {exc}"


def knn_cpu(points: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = cKDTree(points).query(points, k=neighbors + 1)
    return distances[:, 1:], indices[:, 1:]


def knn_gpu(points: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    gpu_points = cp.asarray(points, dtype=cp.float64)
    count = len(points)
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    target_bytes = min(int(free_bytes * 0.15), 512 * 1024**2)
    batch_size = max(1, min(count, target_bytes // max(8 * count, 1)))
    all_distances = np.empty((count, neighbors), dtype=float)
    all_indices = np.empty((count, neighbors), dtype=np.int64)
    point_norms = cp.sum(gpu_points * gpu_points, axis=1)

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        query = gpu_points[start:end]
        squared = (
            cp.sum(query * query, axis=1)[:, None]
            + point_norms[None, :]
            - 2.0 * query @ gpu_points.T
        )
        cp.maximum(squared, 0.0, out=squared)
        squared[cp.arange(end - start), cp.arange(start, end)] = cp.inf
        candidate = cp.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
        candidate_squared = cp.take_along_axis(squared, candidate, axis=1)
        order = cp.argsort(candidate_squared, axis=1)
        candidate = cp.take_along_axis(candidate, order, axis=1)
        candidate_squared = cp.take_along_axis(candidate_squared, order, axis=1)
        all_indices[start:end] = cp.asnumpy(candidate)
        all_distances[start:end] = cp.asnumpy(cp.sqrt(candidate_squared))

    del gpu_points, point_norms
    cp.get_default_memory_pool().free_all_blocks()
    return all_distances, all_indices


def build_knn_graph(
    points: np.ndarray, neighbors: int, use_gpu: bool
):
    if use_gpu:
        distances, indices = knn_gpu(points, neighbors)
    else:
        distances, indices = knn_cpu(points, neighbors)
    rows = np.repeat(np.arange(len(points)), neighbors)
    directed = coo_matrix(
        (distances.reshape(-1), (rows, indices.reshape(-1))),
        shape=(len(points), len(points)),
    ).tocsr()
    directed.sum_duplicates()
    return directed.maximum(directed.T)


def build_cut_surface_graph(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_ids: np.ndarray,
):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = mesh_edges(faces)
    edge_weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    vertex_count = len(vertices)
    sample_nodes = vertex_count + np.arange(len(points), dtype=np.int64)

    sample_vertices = faces[np.asarray(face_ids, dtype=np.int64)]
    sample_rows = np.repeat(sample_nodes, 3)
    sample_cols = sample_vertices.reshape(-1)
    sample_weights = np.linalg.norm(
        np.repeat(points, 3, axis=0) - vertices[sample_cols], axis=1
    )

    rows = np.concatenate([edges[:, 0], sample_rows])
    cols = np.concatenate([edges[:, 1], sample_cols])
    weights = np.concatenate([edge_weights, sample_weights])
    graph = coo_matrix(
        (
            np.concatenate([weights, weights]),
            (
                np.concatenate([rows, cols]),
                np.concatenate([cols, rows]),
            ),
        ),
        shape=(vertex_count + len(points), vertex_count + len(points)),
    ).tocsr()
    graph.sum_duplicates()
    return graph, sample_nodes


def sampled_surface_geodesics(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_ids: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    graph, sample_nodes = build_cut_surface_graph(mesh, points, face_ids)
    result = np.empty((len(points), len(points)), dtype=float)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        distances = dijkstra(
            graph, directed=False, indices=sample_nodes[start:end]
        )
        result[start:end] = distances[:, sample_nodes]
        print(f"  geodesics: {end}/{len(points)}", end="\r", flush=True)
    print()
    if not np.isfinite(result).all():
        raise ValueError("The cut surface graph is disconnected")
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    return result


def classical_mds_cpu(geodesic: np.ndarray) -> np.ndarray:
    squared = geodesic**2
    centered = squared - squared.mean(axis=0) - squared.mean(axis=1)[:, None] + squared.mean()
    gram = -0.5 * centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    if np.any(eigenvalues[order] <= 0.0):
        raise ValueError("Isomap MDS did not produce two positive dimensions")
    return eigenvectors[:, order] * np.sqrt(eigenvalues[order])


def classical_mds_gpu(geodesic: np.ndarray) -> np.ndarray:
    gram = cp.asarray(geodesic, dtype=cp.float64)
    gram *= gram
    row_mean = gram.mean(axis=1)
    col_mean = gram.mean(axis=0)
    total_mean = gram.mean()
    gram -= row_mean[:, None]
    gram -= col_mean[None, :]
    gram += total_mean
    gram *= -0.5
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    order = cp.asarray([len(eigenvalues) - 1, len(eigenvalues) - 2])
    selected_values = eigenvalues[order]
    if bool(cp.any(selected_values <= 0.0).item()):
        raise ValueError("Isomap MDS did not produce two positive dimensions")
    uv = eigenvectors[:, order] * cp.sqrt(selected_values)
    result = cp.asnumpy(uv)
    del gram, eigenvalues, eigenvectors, uv
    cp.get_default_memory_pool().free_all_blocks()
    return result


def run_isomap(
    points: np.ndarray,
    neighbors: int,
    prefer_gpu: bool = True,
    surface_mesh: trimesh.Trimesh | None = None,
    face_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if len(points) < 3:
        raise ValueError("At least three sampled points are required")
    neighbors = min(max(2, neighbors), len(points) - 1)
    available, device_name = gpu_is_available()
    use_gpu = bool(prefer_gpu and available)
    # Standard Isomap uses a graph built directly from the sampled point cloud:
    # k-nearest-neighbor distances -> graph shortest paths -> classical MDS.
    # Do not replace this with the mesh-vertex shortcut graph: that is a custom
    # approximation and can introduce large, topology-dependent distortions.
    print(f"kNN backend: {'GPU' if use_gpu else 'CPU'} (k={neighbors})")
    graph = build_knn_graph(points, neighbors, use_gpu=use_gpu)
    component_count, _ = connected_components(graph, directed=False)
    if component_count != 1:
        raise ValueError(
            f"The kNN graph has {component_count} components; increase --neighbors"
        )
    print("Computing all-pairs graph geodesics on CPU...")
    geodesic = dijkstra(graph, directed=False)
    print(f"MDS backend: {'GPU' if use_gpu else 'CPU'}")
    uv = classical_mds_gpu(geodesic) if use_gpu else classical_mds_cpu(geodesic)
    backend = f"GPU ({device_name}) + CPU Dijkstra" if use_gpu else "CPU"
    return uv, backend


def run_arap_parameterization(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_ids: np.ndarray,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, str]:
    """Flatten the cut triangle mesh, then interpolate sample UV per face."""
    module_path = Path(__file__).with_name("LSCM .py")
    spec = importlib.util.spec_from_file_location("spraycode_lscm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the ARAP parameterizer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.solve_lscm_arap(mesh, iterations=20, prefer_gpu=prefer_gpu)

    triangles = np.asarray(mesh.triangles, dtype=float)[face_ids]
    a = triangles[:, 0]
    edge0 = triangles[:, 1] - a
    edge1 = triangles[:, 2] - a
    offset = points - a
    d00 = np.einsum("ij,ij->i", edge0, edge0)
    d01 = np.einsum("ij,ij->i", edge0, edge1)
    d11 = np.einsum("ij,ij->i", edge1, edge1)
    d20 = np.einsum("ij,ij->i", offset, edge0)
    d21 = np.einsum("ij,ij->i", offset, edge1)
    denominator = d00 * d11 - d01 * d01
    if np.any(np.abs(denominator) <= np.finfo(float).eps):
        raise ValueError("Cannot interpolate UV on a degenerate triangle")
    w1 = (d11 * d20 - d01 * d21) / denominator
    w2 = (d00 * d21 - d01 * d20) / denominator
    w0 = 1.0 - w1 - w2
    face_uv = result.uv[np.asarray(mesh.faces, dtype=np.int64)[face_ids]]
    uv = w0[:, None] * face_uv[:, 0] + w1[:, None] * face_uv[:, 1] + w2[:, None] * face_uv[:, 2]
    backend = (
        f"LSCM + ARAP ({result.compute_backend}, {result.flipped_faces} flipped faces, "
        f"mean edge stretch {result.mean_edge_stretch:.3g})"
    )
    return uv, backend


def run_harmonic_parameterization(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_ids: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Injective disk parameterization with an arc-length convex boundary."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    topology = mesh_topology(mesh)
    if topology["boundaries"] != 1 or topology["euler"] != 1 or topology["genus"] != 0:
        genus = topology["genus"]
        extra_cuts = 2 * genus if genus is not None else "unknown"
        raise ValueError(
            "The seam did not create a topological disk, so a single non-overlapping "
            f"flattening is impossible: {topology_summary(topology)}. This mesh needs "
            f"at least {extra_cuts} additional handle cuts or multiple UV islands. "
            "If genus=None, repair non-manifold topology or disconnected shells first. "
            "--embedding isomap only provides a distance projection; it is not an unfolding."
        )
    loops = ordered_boundary_loops(faces)
    if len(loops) != 1:
        raise ValueError(f"Harmonic flattening needs one boundary loop, found {len(loops)}")
    boundary = loops[0]
    boundary_xyz = vertices[boundary]
    closed_xyz = np.vstack([boundary_xyz, boundary_xyz[0]])
    segment_lengths = np.linalg.norm(np.diff(closed_xyz, axis=0), axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= np.finfo(float).eps:
        raise ValueError("The cut boundary has zero length")
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths[:-1])])
    angle = 2.0 * np.pi * arc / perimeter
    radius = perimeter / (2.0 * np.pi)

    uv_vertices = np.zeros((len(vertices), 2), dtype=float)
    uv_vertices[boundary] = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    boundary_mask = np.zeros(len(vertices), dtype=bool)
    boundary_mask[boundary] = True
    interior = np.flatnonzero(~boundary_mask)

    adjacency: list[set[int]] = [set() for _ in range(len(vertices))]
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
    interior_row = {int(vertex): row for row, vertex in enumerate(interior)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros((len(interior), 2), dtype=float)
    for row, vertex in enumerate(interior):
        neighbors = adjacency[int(vertex)]
        if not neighbors:
            raise ValueError(f"Isolated mesh vertex {vertex}")
        rows.append(row)
        cols.append(row)
        values.append(float(len(neighbors)))
        for neighbor in neighbors:
            if boundary_mask[neighbor]:
                rhs[row] += uv_vertices[neighbor]
            else:
                rows.append(row)
                cols.append(interior_row[neighbor])
                values.append(-1.0)
    if len(interior):
        laplacian = coo_matrix(
            (values, (rows, cols)), shape=(len(interior), len(interior))
        ).tocsr()
        uv_vertices[interior, 0] = spsolve(laplacian, rhs[:, 0])
        uv_vertices[interior, 1] = spsolve(laplacian, rhs[:, 1])

    face_uv = uv_vertices[faces[face_ids]]
    triangles = np.asarray(mesh.triangles, dtype=float)[face_ids]
    a = triangles[:, 0]
    e0 = triangles[:, 1] - a
    e1 = triangles[:, 2] - a
    offset = points - a
    d00 = np.einsum("ij,ij->i", e0, e0)
    d01 = np.einsum("ij,ij->i", e0, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d20 = np.einsum("ij,ij->i", offset, e0)
    d21 = np.einsum("ij,ij->i", offset, e1)
    denominator = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / denominator
    w2 = (d00 * d21 - d01 * d20) / denominator
    w0 = 1.0 - w1 - w2
    uv = w0[:, None] * face_uv[:, 0] + w1[:, None] * face_uv[:, 1] + w2[:, None] * face_uv[:, 2]

    uv_edge0 = uv_vertices[faces[:, 1]] - uv_vertices[faces[:, 0]]
    uv_edge1 = uv_vertices[faces[:, 2]] - uv_vertices[faces[:, 0]]
    signed = uv_edge0[:, 0] * uv_edge1[:, 1] - uv_edge0[:, 1] * uv_edge1[:, 0]
    orientation = np.sign(np.median(signed[signed != 0.0]))
    flipped = int(np.count_nonzero(signed * orientation <= 1.0e-12))
    return uv, f"convex-boundary harmonic map ({flipped} flipped/degenerate faces)"


def save_csv(path: str | Path, points: np.ndarray, uv: np.ndarray, face_ids: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "face_index", "u", "v", "x", "y", "z"])
        for i, (point, coord, face_id) in enumerate(zip(points, uv, face_ids)):
            writer.writerow([i, int(face_id), coord[0], coord[1], *point])


def plot_result(
    points: np.ndarray,
    uv: np.ndarray,
    boundary_mask: np.ndarray | None = None,
    minimum_directions: np.ndarray | None = None,
    gradient_arrows: int = 150,
    save_path: str = "",
    show: bool = True,
) -> None:
    color = uv[:, 0]
    figure = plt.figure(figsize=(12, 5))
    ax3d = figure.add_subplot(1, 2, 1, projection="3d")
    ax3d.scatter(*points.T, c=color, cmap="viridis", s=4)
    if boundary_mask is not None and minimum_directions is not None:
        boundary_indices = np.flatnonzero(boundary_mask)
        if len(boundary_indices):
            ax3d.scatter(*points[boundary_indices].T, c="black", s=8, label="cloud boundary")
            if gradient_arrows > 0 and len(boundary_indices) > gradient_arrows:
                selection = np.linspace(
                    0, len(boundary_indices) - 1, gradient_arrows, dtype=np.int64
                )
                arrow_indices = boundary_indices[selection]
            else:
                arrow_indices = boundary_indices
            extent = float(np.linalg.norm(np.ptp(points, axis=0)))
            arrow_length = 0.035 * max(extent, np.finfo(float).eps)
            ax3d.quiver(
                points[arrow_indices, 0], points[arrow_indices, 1], points[arrow_indices, 2],
                minimum_directions[arrow_indices, 0],
                minimum_directions[arrow_indices, 1],
                minimum_directions[arrow_indices, 2],
                length=arrow_length, normalize=False, color="crimson", linewidth=0.8,
            )
            ax3d.legend(loc="best")
    ax3d.set_title("Point-cloud boundary + minimum-gradient directions")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    ax2d = figure.add_subplot(1, 2, 2)
    ax2d.scatter(uv[:, 0], uv[:, 1], c=color, cmap="viridis", s=4)
    ax2d.set_title("2D parameterization")
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")
    ax2d.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)


def select_stl_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("No STL path was provided and tkinter is unavailable") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an STL model for Isomap",
        filetypes=[("STL models", "*.stl"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stl", nargs="?", default="",
        help="Input STL file; when omitted, a file dialog opens",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Samples per square model unit (default: 1.0)",
    )
    parser.add_argument("--neighbors", type=int, default=12, help="Isomap k-neighbor count")
    parser.add_argument(
        "--embedding", choices=("isomap", "arap", "harmonic"), default="isomap",
        help="2D method: Isomap (default), free-boundary ARAP, or circular harmonic map",
    )
    parser.add_argument(
        "--max-points", type=int, default=4000,
        help="Maximum total samples used by full Isomap (default: 4000)",
    )
    parser.add_argument(
        "--boundary-multiplier", type=float, default=5.0,
        help="Boundary density multiplier relative to interior spacing (default: 5)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU kNN and MDS")
    parser.add_argument(
        "--gradient-neighbors", type=int, default=24,
        help="Point-cloud neighbors for boundary and tangent estimation (default: 24)",
    )
    parser.add_argument(
        "--boundary-angle", type=float, default=135.0,
        help="Minimum empty tangent-plane angle for a cloud boundary point, degrees",
    )
    parser.add_argument(
        "--gradient-arrows", type=int, default=150,
        help="Maximum boundary direction arrows to draw; 0 draws all",
    )
    parser.add_argument(
        "--gradient-out", default="point_cloud_boundary_directions.csv",
        help="Output CSV for point-cloud boundary minimum-gradient directions",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", default="isomap_points.csv", help="Output CSV")
    parser.add_argument("--save-figure", default="", help="Optional output PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stl_path = Path(args.stl) if args.stl else select_stl_file()
    if stl_path is None:
        raise SystemExit("No STL model selected")
    print(f"Input STL: {stl_path.resolve()}")
    mesh = load_stl(stl_path)
    boundary_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
    print(f"Boundary loops in input mesh: {boundary_count}")
    seam_points = np.empty((0, 3), dtype=float)
    topology_mesh = None
    # Seam cutting is only needed by the flattening methods. Standard Isomap
    # must operate on the original sampled surface and must not alter it first.
    if args.embedding != "isomap" and boundary_count == 2:
        mesh, seam_points = cut_two_boundary_loft(mesh)
        topology_mesh = mesh
        after_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
        topology = mesh_topology(mesh)
        print(
            "Two-boundary loft detected: cut along the loft reference curve "
            f"({len(seam_points)} seam vertices)."
        )
        print(f"Boundary loops after cutting: {after_count}")
        print(f"Cut-mesh topology: {topology_summary(topology)}")
    elif args.embedding != "isomap" and boundary_count == 1:
        topology_mesh = mesh
        topology = mesh_topology(mesh)
        print("Single-boundary surface detected: no seam cut is needed.")
        print(f"Mesh topology: {topology_summary(topology)}")
    elif args.embedding == "isomap":
        print("Standard Isomap: no seam cut; sampling the original STL surface.")
    else:
        print("Boundary count is not 2: running without a loft seam.")
    if args.embedding == "harmonic" and topology_mesh is not None:
        topology = mesh_topology(topology_mesh)
        if topology["boundaries"] != 1 or topology["euler"] != 1 or topology["genus"] != 0:
            raise ValueError(
                "Cannot flatten before sampling because the cut mesh is not a valid disk: "
                f"{topology_summary(topology)}. Repair invalid topology and/or add handle "
                "cuts, or use --embedding isomap only as a non-unfolding projection."
            )
    rng = np.random.default_rng(args.seed)
    requested_count = max(3, int(round(float(mesh.area) * args.density)))
    # Dense boundary oversampling is intentionally disabled for standard
    # Isomap, otherwise the point cloud is no longer uniform and the kNN graph
    # becomes density-biased near the boundary.
    boundary_multiplier = args.boundary_multiplier if args.embedding != "isomap" else 0.0
    points, face_ids = sample_surface_uniform_density(
        mesh, args.density, rng, max_points=args.max_points,
        boundary_multiplier=boundary_multiplier,
    )
    print(f"Mesh area: {mesh.area:.6g}")
    print(f"Interior target: {requested_count} ({args.density:g} per square model unit)")
    print(f"Total sampled points (including dense boundary): {len(points)}")
    print(f"Boundary spacing multiplier: {boundary_multiplier:g}x")
    print("Detecting boundary points from point-cloud neighborhoods only ...")
    cloud_boundary_mask = point_cloud_boundary_mask(
        points,
        neighbors=args.gradient_neighbors,
        angle_threshold_degrees=args.boundary_angle,
    )
    normals, minimum_directions, height_gradient_norms, minimum_directional_gradients = (
        estimate_boundary_minimum_gradient_directions(
            points, cloud_boundary_mask, neighbors=args.gradient_neighbors
        )
    )
    print(
        f"Point-cloud boundary samples: {np.count_nonzero(cloud_boundary_mask)} "
        f"(empty-angle threshold {args.boundary_angle:g} degrees)"
    )
    save_boundary_directions(
        args.gradient_out,
        points,
        cloud_boundary_mask,
        normals,
        minimum_directions,
        height_gradient_norms,
        minimum_directional_gradients,
    )
    print(f"Boundary directions saved: {Path(args.gradient_out).resolve()}")
    available, gpu_status = gpu_is_available()
    print(f"CUDA status: {'available' if available else 'unavailable'} ({gpu_status})")
    if args.embedding in ("harmonic", "arap"):
        if topology_mesh is None:
            raise ValueError(
                "Mesh flattening requires a single-boundary disk or a two-boundary loft "
                "that can be cut to a disk; use --embedding isomap for other topology."
            )
        if args.embedding == "harmonic":
            print("Running overlap-free harmonic mesh flattening ...")
            uv, backend = run_harmonic_parameterization(topology_mesh, points, face_ids)
        else:
            print("Running LSCM + ARAP mesh flattening ...")
            uv, backend = run_arap_parameterization(
                topology_mesh, points, face_ids, prefer_gpu=not args.cpu
            )
    else:
        print(f"Running Isomap distance embedding with k={args.neighbors} ...")
        uv, backend = run_isomap(
            points,
            args.neighbors,
            prefer_gpu=not args.cpu,
        )
    print(f"2D backend: {backend}")
    save_csv(args.out, points, uv, face_ids)
    print(f"Saved: {Path(args.out).resolve()}")
    if not args.no_show or args.save_figure:
        if args.no_show:
            plt.switch_backend("Agg")
        plot_result(
            points,
            uv,
            cloud_boundary_mask,
            minimum_directions,
            args.gradient_arrows,
            args.save_figure,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
