#!/usr/bin/env python3
"""Load an STL surface or PLY point cloud and embed it in 2D with Isomap.

Dependencies:
    py -m pip install numpy scipy trimesh matplotlib cupy-cuda11x
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Keep generated Matplotlib/CuPy caches inside this workspace.  ``parents[2]``
# resolves to the drive root for PCLTrajPlan/trajplanpcl.py and makes importing
# this module require write access to D:\, which is neither necessary nor safe.
RUNTIME_CACHE = Path(__file__).resolve().parents[1] / ".runtime_cache"
RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(RUNTIME_CACHE / "cupy"))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, cKDTree

try:
    import cupy as cp
except Exception:
    cp = None


def load_stl(path: str | Path) -> trimesh.Trimesh:#加载并预处理网格
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
    # improves downstream boundary analysis and UV diagnostics.
    trimesh.repair.fix_winding(loaded)
    return loaded


_PLY_SCALAR_DTYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "int64": "i8", "uint64": "u8",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def _ply_numpy_dtype(type_name: str, endian: str) -> np.dtype:
    code = _PLY_SCALAR_DTYPES.get(type_name.lower())
    if code is None:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return np.dtype(endian + code)


def _read_ply_header(stream) -> tuple[str, list[dict]]:
    """Read a PLY header without interpreting unrelated data elements."""
    if stream.readline().strip().lower() != b"ply":
        raise ValueError("The selected file is not a PLY file")

    format_name = ""
    elements: list[dict] = []
    current: dict | None = None
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PLY header is not terminated by end_header")
        line = raw.decode("ascii", errors="replace").strip().split()
        if not line:
            continue
        keyword = line[0].lower()
        if keyword == "end_header":
            break
        if keyword == "format" and len(line) >= 2:
            format_name = line[1].lower()
        elif keyword == "element" and len(line) == 3:
            current = {
                "name": line[1].lower(),
                "count": int(line[2]),
                "properties": [],
            }
            elements.append(current)
        elif keyword == "property":
            if current is None:
                raise ValueError("PLY property appears before any element")
            if len(line) == 3:
                current["properties"].append(
                    ("scalar", line[1].lower(), "", line[2].lower())
                )
            elif len(line) == 5 and line[1].lower() == "list":
                current["properties"].append(
                    ("list", line[2].lower(), line[3].lower(), line[4].lower())
                )
            else:
                raise ValueError(f"Unsupported PLY property declaration: {' '.join(line)}")

    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported or missing PLY format: {format_name}")
    return format_name, elements


def _skip_binary_ply_element(stream, element: dict, endian: str) -> None:
    """Skip an element preceding vertex, including variable-length lists."""
    properties = element["properties"]
    if not properties:
        if element["count"] != 0:
            raise ValueError(f"PLY element {element['name']} has data but no properties")
        return
    if all(prop[0] == "scalar" for prop in properties):
        record_size = sum(_ply_numpy_dtype(prop[1], endian).itemsize for prop in properties)
        stream.seek(element["count"] * record_size, 1)
        return

    for _ in range(element["count"]):
        for kind, first_type, second_type, _name in properties:
            if kind == "scalar":
                stream.seek(_ply_numpy_dtype(first_type, endian).itemsize, 1)
                continue
            count_dtype = _ply_numpy_dtype(first_type, endian)
            raw_count = stream.read(count_dtype.itemsize)
            if len(raw_count) != count_dtype.itemsize:
                raise ValueError("PLY file ended inside a list property")
            list_size = int(np.frombuffer(raw_count, dtype=count_dtype, count=1)[0])
            stream.seek(list_size * _ply_numpy_dtype(second_type, endian).itemsize, 1)


def load_ply_points(path: str | Path) -> np.ndarray:
    """Load only PLY XYZ vertices, ignoring incompatible auxiliary elements."""
    with Path(path).open("rb") as stream:
        format_name, elements = _read_ply_header(stream)
        vertex_position = next(
            (index for index, element in enumerate(elements) if element["name"] == "vertex"),
            None,
        )
        if vertex_position is None:
            raise ValueError(f"No vertex element found in {path}")
        vertex = elements[vertex_position]
        if vertex["count"] < 3:
            raise ValueError("The PLY file must contain at least three vertices")

        if format_name == "ascii":
            for element in elements[:vertex_position]:
                for _ in range(element["count"]):
                    if not stream.readline():
                        raise ValueError("PLY file ended before vertex data")
            if any(prop[0] != "scalar" for prop in vertex["properties"]):
                raise ValueError("List properties inside PLY vertices are unsupported")
            names = [prop[3] for prop in vertex["properties"]]
            try:
                xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
            except ValueError as exc:
                raise ValueError("PLY vertex element does not contain x, y and z") from exc
            points = np.empty((vertex["count"], 3), dtype=float)
            for row in range(vertex["count"]):
                values = stream.readline().split()
                if len(values) < len(names):
                    raise ValueError("PLY ASCII vertex row has too few values")
                points[row] = [float(values[index]) for index in xyz_indices]
        else:
            endian = "<" if format_name == "binary_little_endian" else ">"
            for element in elements[:vertex_position]:
                _skip_binary_ply_element(stream, element, endian)
            if any(prop[0] != "scalar" for prop in vertex["properties"]):
                raise ValueError("List properties inside PLY vertices are unsupported")
            names = [prop[3] for prop in vertex["properties"]]
            try:
                xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
            except ValueError as exc:
                raise ValueError("PLY vertex element does not contain x, y and z") from exc
            record_dtype = np.dtype([
                (f"field_{index}", _ply_numpy_dtype(prop[1], endian))
                for index, prop in enumerate(vertex["properties"])
            ])
            vertex_data = np.fromfile(stream, dtype=record_dtype, count=vertex["count"])
            if len(vertex_data) != vertex["count"]:
                raise ValueError("PLY binary vertex data is truncated")
            points = np.column_stack([
                vertex_data[f"field_{index}"].astype(float, copy=False)
                for index in xyz_indices
            ])

    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 3:
        raise ValueError("The PLY file must contain at least three XYZ points")
    if not np.all(np.isfinite(points)):
        raise ValueError("The PLY point cloud contains non-finite coordinates")
    return points


def limit_points_uniformly(
    points: np.ndarray, max_points: int, seed: int
) -> np.ndarray:
    """Uniformly sample input rows without replacement for full Isomap."""
    if max_points <= 0 or len(points) <= max_points:
        return points
    target_count = max(3, int(max_points))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(points), size=target_count, replace=False)
    # Sorting retains the original file order while selection remains random.
    selected.sort()
    return points[selected]


def ordered_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:#边界识别
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


def estimate_global_minimum_gradient(
    points: np.ndarray, neighbors: int = 24, prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Estimate a local field on all points and its global optimum direction.

    For every point, local PCA gives a normal n and the tangent-plane height
    gradient g.  The global direction minimizes mean (g dot d)^2, hence is the
    smallest-eigenvalue eigenvector of mean(g g^T), as in groupgrid.py.
    """
    if len(points) < 4:
        raise ValueError("At least four points are required")
    neighbors = min(max(3, neighbors), len(points) - 1)
    # Neighborhood lookup is memory-safe on CPU; the expensive batched eigens
    # and vector operations below use CuPy when requested and available.
    _, neighborhood_indices = cKDTree(points).query(points, k=neighbors + 1)
    local = points[neighborhood_indices[:, 1:]]
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighbors - 1, 1)
    available, device_name = gpu_is_available()
    use_gpu = bool(prefer_gpu and available)
    xp = cp if use_gpu else np
    covariance_xp = xp.asarray(covariance)
    _, eigenvectors = xp.linalg.eigh(covariance_xp)
    normals = eigenvectors[:, :, 0]
    tangent_fallback = eigenvectors[:, :, 2]
    vertical = xp.asarray([0.0, 0.0, 1.0])
    gradients = vertical - xp.sum(normals * vertical, axis=1, keepdims=True) * normals
    gradient_norms = xp.linalg.norm(gradients, axis=1)
    directions = xp.cross(normals, vertical)
    direction_norms = xp.linalg.norm(directions, axis=1)
    valid = direction_norms > 1.0e-10
    directions = xp.where(
        valid[:, None], directions / xp.maximum(direction_norms[:, None], 1.0e-12),
        tangent_fallback,
    )
    directions /= xp.maximum(xp.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    min_values = xp.abs(xp.sum(gradients * directions, axis=1))
    gradient_scatter = gradients.T @ gradients / len(points)
    global_values, global_vectors = xp.linalg.eigh(gradient_scatter)
    global_direction = global_vectors[:, 0]
    if float(global_direction[2].item()) < 0.0:
        global_direction = -global_direction
    to_cpu = lambda value: cp.asnumpy(value) if use_gpu else np.asarray(value)
    backend = f"GPU ({device_name})" if use_gpu else "CPU"
    return (
        to_cpu(normals), to_cpu(directions), to_cpu(gradient_norms),
        to_cpu(min_values), to_cpu(global_direction), backend,
    )


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
    # CuPy's default non-blocking NumPy transfer uses a pinned-memory staging
    # buffer.  On some Windows CUDA drivers this can fail with
    # cudaErrorAlreadyMapped.  A blocking transfer from pageable memory avoids
    # that resource-mapping path.
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.set_pinned_memory_allocator(None)
    try:
        gram = cp.asarray(geodesic, dtype=cp.float64, blocking=True)
    except TypeError:
        # Compatibility with CuPy versions predating the blocking keyword.
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
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return result


def run_isomap(
    points: np.ndarray,
    neighbors: int,
    prefer_gpu: bool = True,
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
    mds_gpu_used = use_gpu
    if use_gpu:
        try:
            uv = classical_mds_gpu(geodesic)
        except Exception as exc:
            # A CUDA transfer, allocation, or eigensolver failure must not
            # discard the already-computed geodesic matrix.  Release cached
            # CUDA memory and finish MDS on CPU instead.
            mds_gpu_used = False
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            print(f"Warning: GPU MDS failed ({type(exc).__name__}: {exc})")
            print("MDS backend: CPU fallback")
            uv = classical_mds_cpu(geodesic)
    else:
        uv = classical_mds_cpu(geodesic)
    if use_gpu and mds_gpu_used:
        backend = f"GPU ({device_name}) + CPU Dijkstra"
    elif use_gpu:
        backend = f"GPU kNN ({device_name}) + CPU Dijkstra + CPU MDS fallback"
    else:
        backend = "CPU"
    return uv, backend


def save_csv(path: str | Path, points: np.ndarray, uv: np.ndarray, face_ids: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "face_index", "u", "v", "x", "y", "z"])
        for i, (point, coord, face_id) in enumerate(zip(points, uv, face_ids)):
            writer.writerow([i, int(face_id), coord[0], coord[1], *point])


def estimate_uv_drive_direction(
    points: np.ndarray,
    uv: np.ndarray,
    boundary_mask: np.ndarray,
    minimum_directions: np.ndarray,
    neighbors: int = 24,
) -> np.ndarray:
    """Map 3-D minimum-gradient tangents to one robust 2-D travel direction.

    A local 3-D displacement parallel to each boundary minimum-gradient vector
    is paired with the corresponding displacement in UV.  The dominant
    unoriented UV line is obtained from a 2-D scatter matrix, so arbitrary
    signs of local vectors cannot cancel the result.
    """
    indices = np.flatnonzero(boundary_mask)
    if len(indices) < 2:
        centered = uv - uv.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return vh[0] / max(np.linalg.norm(vh[0]), np.finfo(float).eps)
    tree = cKDTree(points)
    k = min(max(4, neighbors), len(points))
    uv_vectors: list[np.ndarray] = []
    for point_index in indices:
        direction = minimum_directions[point_index]
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1.0e-12:
            continue
        _, local_indices = tree.query(points[point_index], k=k)
        local_indices = np.atleast_1d(local_indices)[1:]
        displacement = points[local_indices] - points[point_index]
        lengths = np.linalg.norm(displacement, axis=1)
        valid = lengths > 1.0e-12
        if not np.any(valid):
            continue
        displacement = displacement[valid]
        local_indices = local_indices[valid]
        alignment = np.abs((displacement / np.linalg.norm(displacement, axis=1)[:, None]) @
                           (direction / direction_norm))
        selected = int(np.argmax(alignment))
        uv_delta = uv[local_indices[selected]] - uv[point_index]
        uv_norm = np.linalg.norm(uv_delta)
        if uv_norm > 1.0e-12:
            uv_vectors.append(uv_delta / uv_norm)
    if not uv_vectors:
        _, _, vh = np.linalg.svd(uv - uv.mean(axis=0), full_matrices=False)
        return vh[0] / max(np.linalg.norm(vh[0]), np.finfo(float).eps)
    vectors = np.asarray(uv_vectors)
    scatter = vectors.T @ vectors / len(vectors)
    _, eigenvectors = np.linalg.eigh(scatter)
    travel = eigenvectors[:, -1]
    return travel / max(np.linalg.norm(travel), np.finfo(float).eps)


def project_global_direction_to_uv(
    points: np.ndarray,
    uv: np.ndarray,
    boundary_mask: np.ndarray,
    global_direction: np.ndarray,
    neighbors: int = 24,
) -> np.ndarray:
    """Map one global 3-D direction to a robust direction in UV.

    At each boundary sample, the neighboring 3-D displacement most parallel
    to ``global_direction`` is paired with its corresponding UV displacement.
    The UV lines are combined in a weighted scatter matrix, with stronger 3-D
    alignment receiving more weight.  This keeps the mapping local while the
    direction being mapped is explicitly the global minimum-gradient vector.
    """
    direction = np.asarray(global_direction, dtype=float).reshape(-1)
    if direction.shape != (3,):
        raise ValueError("global_direction must contain exactly three values")
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 1.0e-12:
        raise ValueError("global_direction must be non-zero")
    direction /= direction_norm

    indices = np.flatnonzero(boundary_mask)
    if len(indices) < 2:
        # A closed point cloud may have no detected boundary.  Use every point
        # while retaining the same local correspondence mapping.
        indices = np.arange(len(points), dtype=np.int64)

    tree = cKDTree(points)
    k = min(max(4, neighbors), len(points))
    uv_vectors: list[np.ndarray] = []
    weights: list[float] = []
    oriented_vectors: list[np.ndarray] = []
    for point_index in indices:
        _, local_indices = tree.query(points[point_index], k=k)
        local_indices = np.atleast_1d(local_indices)
        local_indices = local_indices[local_indices != point_index]
        displacement = points[local_indices] - points[point_index]
        lengths = np.linalg.norm(displacement, axis=1)
        valid = lengths > 1.0e-12
        if not np.any(valid):
            continue
        displacement = displacement[valid]
        local_indices = local_indices[valid]
        signed_alignment = (displacement / lengths[valid, None]) @ direction
        selected = int(np.argmax(np.abs(signed_alignment)))
        uv_delta = uv[local_indices[selected]] - uv[point_index]
        uv_norm = np.linalg.norm(uv_delta)
        if uv_norm <= 1.0e-12:
            continue

        uv_unit = uv_delta / uv_norm
        alignment = float(signed_alignment[selected])
        weight = max(abs(alignment), np.finfo(float).eps) ** 2
        uv_vectors.append(uv_unit)
        weights.append(weight)
        # Give the final eigenvector a reproducible orientation corresponding
        # to the positive global_direction (trajectory geometry itself is an
        # unoriented line, but stable output is useful for logs and plotting).
        oriented_vectors.append(uv_unit if alignment >= 0.0 else -uv_unit)

    if not uv_vectors:
        # Degenerate-neighborhood fallback: a global least-squares affine map
        # still uses all points<->UV correspondences rather than dropping the
        # third coordinate or reverting to local minimum_directions.
        centered_points = points - points.mean(axis=0)
        centered_uv = uv - uv.mean(axis=0)
        affine_map, _, _, _ = np.linalg.lstsq(centered_points, centered_uv, rcond=None)
        travel = direction @ affine_map
        travel_norm = np.linalg.norm(travel)
        if travel_norm <= 1.0e-12:
            raise ValueError("Cannot map global_direction into the UV domain")
        return travel / travel_norm

    vectors = np.asarray(uv_vectors)
    weight_array = np.asarray(weights)
    scatter = (vectors * weight_array[:, None]).T @ vectors / weight_array.sum()
    _, eigenvectors = np.linalg.eigh(scatter)
    travel = eigenvectors[:, -1]
    oriented_mean = np.average(np.asarray(oriented_vectors), axis=0, weights=weight_array)
    if travel @ oriented_mean < 0.0:
        travel = -travel
    return travel / max(np.linalg.norm(travel), np.finfo(float).eps)


def line_convex_polygon_intersection(
    center: np.ndarray, direction: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Intersect an infinite line with a convex polygon."""
    normal = np.array([-direction[1], direction[0]])
    values = (polygon - center) @ normal
    intersections: list[np.ndarray] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        a, b = values[index], values[(index + 1) % len(polygon)]
        if abs(a) <= 1.0e-10:
            intersections.append(start)
        if a * b < -1.0e-20:
            fraction = a / (a - b)
            intersections.append(start + fraction * (end - start))
    if len(intersections) < 2:
        return None
    candidates = np.unique(np.round(np.asarray(intersections), decimals=12), axis=0)
    if len(candidates) < 2:
        return None
    order = np.argsort(candidates @ direction)
    return candidates[order[0]], candidates[order[-1]]


def generate_coverage_trajectory(
    uv: np.ndarray,
    travel_direction: np.ndarray,
    spacing: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Generate boustrophedon, equal-spacing lines over the UV convex hull."""
    if len(uv) < 3:
        raise ValueError("At least three UV points are required for trajectory planning")
    travel = travel_direction / max(np.linalg.norm(travel_direction), np.finfo(float).eps)
    normal = np.array([-travel[1], travel[0]])
    hull = uv[ConvexHull(uv).vertices]
    projections = hull @ normal
    span = float(projections.max() - projections.min())
    if span <= 1.0e-12:
        raise ValueError("The UV domain has zero width perpendicular to travel direction")
    if spacing <= 0.0:
        spacing = span / 40.0
    if spacing <= 0.0:
        raise ValueError("Trajectory spacing must be positive")
    line_count = max(2, int(np.floor(span / spacing)) + 1)
    offsets = np.linspace(projections.min(), projections.max(), line_count)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in offsets:
        center = normal * offset
        segment = line_convex_polygon_intersection(center, travel, hull)
        if segment is not None and np.linalg.norm(segment[1] - segment[0]) > 1.0e-12:
            segments.append(segment)
    if not segments:
        raise ValueError("No coverage lines intersect the UV domain")
    trajectory: list[np.ndarray] = []
    for line_index, (start, end) in enumerate(segments):
        if line_index % 2:
            start, end = end, start
        # Consecutive end/start pairs form the short boundary connector.
        trajectory.extend((start, end))
    return np.asarray(trajectory), float(spacing)


def densify_polyline(polyline: np.ndarray, step: float) -> np.ndarray:
    """Insert equal-distance samples so the inverse-mapped 3-D path is smooth."""
    if len(polyline) < 2:
        return polyline.copy()
    dense: list[np.ndarray] = []
    for start, end in zip(polyline[:-1], polyline[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / max(step, 1.0e-12))))
        for fraction in np.arange(count, dtype=float) / count:
            dense.append(start + fraction * (end - start))
    dense.append(polyline[-1])
    return np.asarray(dense)


def inverse_map_uv_to_xyz(
    trajectory_uv: np.ndarray,
    sample_uv: np.ndarray,
    sample_xyz: np.ndarray,
    neighbors: int = 12,
) -> np.ndarray:
    """Approximate Isomap's inverse with weighted local affine regression."""
    k = min(max(3, neighbors), len(sample_uv))
    distances, indices = cKDTree(sample_uv).query(trajectory_uv, k=k)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    if distances.shape[0] != len(trajectory_uv):
        distances = distances.T
        indices = indices.T
    result = np.empty((len(trajectory_uv), 3), dtype=float)
    for row, query in enumerate(trajectory_uv):
        local_uv = sample_uv[indices[row]]
        local_xyz = sample_xyz[indices[row]]
        scale = max(float(np.median(distances[row])), 1.0e-12)
        weights = 1.0 / (1.0 + (distances[row] / scale) ** 2)
        weight_sum = float(weights.sum())
        uv_center = np.sum(weights[:, None] * local_uv, axis=0) / weight_sum
        xyz_center = np.sum(weights[:, None] * local_xyz, axis=0) / weight_sum
        du = local_uv - uv_center
        dx = local_xyz - xyz_center
        normal_matrix = du.T @ (weights[:, None] * du)
        regularization = 1.0e-8 * max(float(np.trace(normal_matrix)), 1.0)
        jacobian = np.linalg.solve(
            normal_matrix + regularization * np.eye(2),
            du.T @ (weights[:, None] * dx),
        )
        result[row] = xyz_center + (query - uv_center) @ jacobian
    return result


def save_trajectory_csv(
    path: str | Path, trajectory_uv: np.ndarray, trajectory_xyz: np.ndarray
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trajectory_index", "u", "v", "x", "y", "z"])
        for index, (uv_point, xyz_point) in enumerate(zip(trajectory_uv, trajectory_xyz)):
            writer.writerow([index, *uv_point, *xyz_point])


def plot_coverage_trajectory(
    uv: np.ndarray, trajectory: np.ndarray, travel: np.ndarray,
    spacing: float, output: str | Path, show: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(uv[:, 0], uv[:, 1], s=5, c="0.78", alpha=0.55, label="2-D point cloud")
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", linewidth=1.4,
              label=f"coverage path (spacing={spacing:.4g})")
    center = uv.mean(axis=0)
    arrow_length = 0.12 * max(np.ptp(uv, axis=0).max(), 1.0)
    axis.arrow(center[0], center[1], travel[0] * arrow_length, travel[1] * arrow_length,
               width=0.002 * arrow_length, color="tab:blue", length_includes_head=True,
               label="minimum-gradient travel direction")
    axis.set_title("Equal-spacing coverage trajectory in Isomap UV")
    axis.set_xlabel("u")
    axis.set_ylabel("v")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_combined_result(
    points: np.ndarray,
    uv: np.ndarray,
    trajectory_uv: np.ndarray,
    trajectory_xyz: np.ndarray,
    local_directions: np.ndarray,
    global_direction: np.ndarray,
    travel_direction: np.ndarray,
    boundary_mask: np.ndarray,
    spacing: float,
    output: str | Path,
    show: bool,
    arrows: int = 180,
) -> None:
    """Show 3-D local/global directions, UV cloud, and UV coverage together."""
    figure = plt.figure(figsize=(19, 6), constrained_layout=True)
    ax3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax3d.scatter(*points.T, s=3, c="0.72", alpha=0.28, depthshade=False)
    selected = np.linspace(0, len(points) - 1, min(arrows, len(points)), dtype=np.int64)
    arrow_length = 0.045 * max(float(np.ptp(points, axis=0).max()), 1.0)
    ax3d.quiver(
        points[selected, 0], points[selected, 1], points[selected, 2],
        local_directions[selected, 0], local_directions[selected, 1],
        local_directions[selected, 2], length=arrow_length, normalize=True,
        color="tab:orange", linewidth=0.65, alpha=0.65,
        label="local minimum-gradient",
    )
    center = points.mean(axis=0)
    ax3d.quiver(
        *center, *global_direction, length=3.0 * arrow_length, normalize=True,
        color="red", linewidth=3.0, arrow_length_ratio=0.2,
        label="global minimum-gradient",
    )
    ax3d.plot(
        trajectory_xyz[:, 0], trajectory_xyz[:, 1], trajectory_xyz[:, 2],
        color="magenta", linewidth=1.4, alpha=0.9,
        label="inverse-mapped coverage path",
    )
    if np.any(boundary_mask):
        ax3d.scatter(*points[boundary_mask].T, s=8, c="black", alpha=0.7, label="boundary")
    ax3d.set_title("3-D point cloud + gradient directions")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-12))
    ax3d.legend(fontsize=8, loc="best")

    axuv = figure.add_subplot(1, 3, 2)
    axuv.scatter(uv[:, 0], uv[:, 1], s=5, c="0.65", alpha=0.65)
    axuv.set_title("Isomap 2-D point cloud")
    axuv.set_xlabel("u")
    axuv.set_ylabel("v")
    axuv.set_aspect("equal", adjustable="datalim")
    axuv.grid(alpha=0.2)

    axpath = figure.add_subplot(1, 3, 3)
    axpath.scatter(uv[:, 0], uv[:, 1], s=5, c="0.78", alpha=0.5, label="2-D points")
    axpath.plot(trajectory_uv[:, 0], trajectory_uv[:, 1], c="tab:red", lw=1.3,
                label=f"coverage path (spacing={spacing:.4g})")
    travel = travel_direction / max(
        np.linalg.norm(travel_direction), np.finfo(float).eps
    )
    center_uv = uv.mean(axis=0)
    arrow_length_uv = 0.12 * max(np.ptp(uv, axis=0).max(), 1.0)
    axpath.arrow(center_uv[0], center_uv[1], travel[0] * arrow_length_uv,
                 travel[1] * arrow_length_uv, width=0.002 * arrow_length_uv,
                 color="tab:blue", length_includes_head=True,
                 label="minimum-gradient travel direction")
    axpath.set_title("Equal-spacing coverage trajectory")
    axpath.set_xlabel("u")
    axpath.set_ylabel("v")
    axpath.set_aspect("equal", adjustable="datalim")
    axpath.grid(alpha=0.2)
    axpath.legend(fontsize=8, loc="best")
    figure.savefig(output, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(figure)


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


def select_input_file() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("No input path was provided and tkinter is unavailable") from exc

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an STL model or PLY point cloud",
        filetypes=[
            ("Supported files", "*.stl *.ply"),
            ("STL models", "*.stl"),
            ("PLY point clouds", "*.ply"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default="",
        help="Input STL or PLY file; when omitted, a file dialog opens",
    )
    parser.add_argument(
        "--density", type=float, default=1.0,
        help="Samples per square model unit (default: 1.0)",
    )
    parser.add_argument("--neighbors", type=int, default=12, help="Isomap k-neighbor count")
    parser.add_argument(
        "--max-points", type=int, default=4000,
        help="Maximum STL/PLY points used by full Isomap; 0 disables cap (default: 4000)",
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
    parser.add_argument(
        "--trajectory-spacing", type=float, default=0.0,
        help="Equal spacing in UV units; 0 chooses domain width / 40",
    )
    parser.add_argument(
        "--trajectory-step", type=float, default=0.0,
        help="UV sampling step before inverse mapping; 0 uses spacing / 4",
    )
    parser.add_argument(
        "--inverse-neighbors", type=int, default=12,
        help="UV neighbors used by local affine inverse mapping (default: 12)",
    )
    parser.add_argument(
        "--trajectory-out", default="coverage_trajectory.csv",
        help="Coverage trajectory CSV output",
    )
    parser.add_argument(
        "--trajectory-figure", default="coverage_trajectory.png",
        help="Coverage trajectory PNG output",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else select_input_file()
    if input_path is None:
        raise SystemExit("No input file selected")
    suffix = input_path.suffix.lower()
    if suffix not in {".stl", ".ply"}:
        raise SystemExit("Input must be an STL model or PLY point cloud")
    print(f"Input {suffix[1:].upper()}: {input_path.resolve()}")

    if suffix == ".ply":
        points = load_ply_points(input_path)
        loaded_count = len(points)
        points = limit_points_uniformly(points, args.max_points, args.seed)
        face_ids = np.full(len(points), -1, dtype=np.int64)
        if len(points) < loaded_count:
            print(
                f"Loaded {loaded_count} PLY points; uniformly random-sampled "
                f"{len(points)} without replacement for full Isomap "
                f"(seed={args.seed})."
            )
        else:
            print(f"Loaded all {loaded_count} PLY points (point cap not reached).")
    else:
        mesh = load_stl(input_path)
        boundary_count = len(ordered_boundary_loops(np.asarray(mesh.faces, dtype=np.int64)))
        print(f"Boundary loops in input mesh: {boundary_count}")
        print("Sampling the STL surface, then continuing with point-cloud Isomap.")
        rng = np.random.default_rng(args.seed)
        requested_count = max(3, int(round(float(mesh.area) * args.density)))
        # Isomap avoids dense boundary sampling to prevent kNN density bias.
        boundary_multiplier = 0.0
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
    normals, minimum_directions, height_gradient_norms, minimum_directional_gradients, global_direction, gradient_backend = (
        estimate_global_minimum_gradient(
            points, neighbors=args.gradient_neighbors, prefer_gpu=not args.cpu
        )
    )
    print(
        f"Point-cloud boundary samples: {np.count_nonzero(cloud_boundary_mask)} "
        f"(empty-angle threshold {args.boundary_angle:g} degrees)"
    )
    print(f"Boundary directions saved: {Path(args.gradient_out).resolve()}")
    print(f"Gradient field backend: {gradient_backend}")
    print(f"Global minimum-gradient direction: {global_direction.tolist()}")
    available, gpu_status = gpu_is_available()
    print(f"CUDA status: {'available' if available else 'unavailable'} ({gpu_status})")
    print(f"Running Isomap distance embedding with k={args.neighbors} ...")
    uv, backend = run_isomap(
        points,
        args.neighbors,
        prefer_gpu=not args.cpu,
    )
    print(f"2D backend: {backend}")
    save_csv(args.out, points, uv, face_ids)
    print(f"Saved: {Path(args.out).resolve()}")

    print("Projecting global minimum-gradient direction to UV ...")
    travel_direction = project_global_direction_to_uv(
        points, uv, cloud_boundary_mask, global_direction,
        neighbors=args.gradient_neighbors,
    )
    trajectory_uv, actual_spacing = generate_coverage_trajectory(
        uv, travel_direction, spacing=args.trajectory_spacing,
    )
    coverage_line_count = len(trajectory_uv) // 2
    trajectory_step = args.trajectory_step if args.trajectory_step > 0.0 else actual_spacing / 4.0
    dense_trajectory_uv = densify_polyline(trajectory_uv, trajectory_step)
    trajectory_xyz = inverse_map_uv_to_xyz(
        dense_trajectory_uv, uv, points, neighbors=args.inverse_neighbors,
    )
    print("Inverse mapping uses point-cloud neighborhoods only (no mesh projection).")
    save_trajectory_csv(args.trajectory_out, dense_trajectory_uv, trajectory_xyz)
    print(f"UV travel direction: {travel_direction.tolist()}")
    print(f"Coverage lines: {coverage_line_count}")
    print(f"Trajectory spacing: {actual_spacing:g}")
    print(f"Inverse-mapped trajectory points: {len(trajectory_xyz)}")
    print(f"Trajectory CSV: {Path(args.trajectory_out).resolve()}")
    combined_figure = args.save_figure or args.trajectory_figure
    plot_combined_result(
        points, uv, dense_trajectory_uv, trajectory_xyz,
        minimum_directions, global_direction, travel_direction,
        cloud_boundary_mask, actual_spacing, combined_figure,
        show=not args.no_show, arrows=args.gradient_arrows,
    )
    print(f"Combined three-panel figure: {Path(combined_figure).resolve()}")


if __name__ == "__main__":
    main()

