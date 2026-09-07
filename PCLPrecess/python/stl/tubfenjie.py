"""官方 OptCuts 的 Python 启动与结果可视化入口。"""

from __future__ import annotations

import os
import heapq
import subprocess
import uuid
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import trimesh


def build_uv_domain(uv: np.ndarray, uv_faces: np.ndarray):
    """Build the valid 2-D parameter domain from UV triangles."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    uv = np.asarray(uv, dtype=float)
    faces = np.asarray(uv_faces, dtype=np.int64)
    if uv.ndim != 2 or uv.shape[1] != 2 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("uv must have shape (N, 2) and uv_faces must have shape (M, 3).")
    triangles = []
    for face in faces:
        points = uv[face]
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > 1.0e-14:
            triangles.append(polygon)
    if not triangles:
        raise ValueError("The UV mesh contains no non-degenerate triangles.")
    domain = unary_union(triangles).buffer(0)
    if domain.is_empty:
        raise ValueError("Could not construct a valid UV domain.")
    return domain


def _largest_axis_aligned_rectangle(domain, uv: np.ndarray, split_line=None):
    """Find a large rectangle fully covered by *domain*.

    Candidate sides come from UV coordinates.  For very dense UV meshes the
    coordinates are reduced to at most 96 quantile samples per axis, keeping
    the search bounded while retaining the exact boundary coordinates at the
    extrema.
    """
    from shapely.geometry import LineString, box

    split_geometry = None
    if split_line is not None:
        split_line = np.asarray(split_line, dtype=float)
        if split_line.ndim != 2 or split_line.shape[1] != 2 or len(split_line) < 2:
            raise ValueError("split_line must have shape (N, 2) with at least two points.")
        split_geometry = LineString(split_line)
        if split_geometry.is_empty or split_geometry.length <= 1.0e-14:
            raise ValueError("split_line must have non-zero length.")

    uv = np.asarray(uv, dtype=float)
    def candidates(values):
        values = np.unique(values[np.isfinite(values)])
        if len(values) <= 96:
            return values
        samples = np.quantile(values, np.linspace(0.0, 1.0, 96))
        return np.unique(samples)
    xs, ys = candidates(uv[:, 0]), candidates(uv[:, 1])
    best = None
    best_area = 0.0
    for left_index in range(len(xs) - 1):
        for right_index in range(left_index + 1, len(xs)):
            width = xs[right_index] - xs[left_index]
            if width * (ys[-1] - ys[0]) <= best_area:
                continue
            for bottom_index in range(len(ys) - 1):
                for top_index in range(bottom_index + 1, len(ys)):
                    area = width * (ys[top_index] - ys[bottom_index])
                    if area <= best_area:
                        continue
                    candidate = box(xs[left_index], ys[bottom_index], xs[right_index], ys[top_index])
                    if domain.covers(candidate) and (
                        split_geometry is None
                        or candidate.boundary.intersects(split_geometry)
                    ):
                        best, best_area = candidate, area
    if best is None:
        raise ValueError("Could not find an interior rectangle in the UV domain.")
    return best


def decompose_uv_domain(
    uv: np.ndarray, uv_faces: np.ndarray, split_line: np.ndarray | None = None
) -> dict[str, object]:
    """Return the maximum covered rectangle and non-overlapping remainder triangles.

    The rectangle is selected first and is excluded from the remainder before
    triangulation.  Every returned triangle is fully covered by the original
    UV domain and lies outside the rectangle (within numerical tolerance).
    """
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    uv = np.asarray(uv, dtype=float)
    domain = build_uv_domain(uv, uv_faces)
    rectangle = _largest_axis_aligned_rectangle(domain, uv, split_line=split_line)
    remainder = domain.difference(rectangle).buffer(0)
    triangles = []
    for piece in getattr(remainder, "geoms", [remainder]):
        if piece.is_empty or piece.area <= 1.0e-14:
            continue
        for triangle in triangulate(piece):
            if triangle.area <= 1.0e-14 or not piece.covers(triangle):
                continue
            coordinates = np.asarray(triangle.exterior.coords[:-1], dtype=float)
            if len(coordinates) == 3:
                triangles.append(coordinates)
    return {
        "domain": domain,
        "rectangle": np.asarray(rectangle.exterior.coords[:-1], dtype=float),
        "rectangle_bounds": tuple(float(value) for value in rectangle.bounds),
        "split_line": None if split_line is None else np.asarray(split_line, dtype=float),
        "remainder": remainder,
        "triangles": triangles,
    }


# OptCut.py lives in ``PCLPrecess/python/stl``.  The OptCuts checkout is a
# sibling of ``python`` under ``PCLPrecess``, so parents[2] is the project
# root (parents[1] would incorrectly point at ``PCLPrecess/python``).
ROOT = Path(__file__).resolve().parents[2]
OPTCUTS_ROOT = ROOT / "third_party" / "OptCuts"
OPTCUTS_EXE = OPTCUTS_ROOT / "build" / "Release" / "OptCuts_bin.exe"
FINAL_FRAME_HOLD_MS = 2000


def ordered_boundary_loops(faces: np.ndarray) -> List[np.ndarray]:
    """Return the vertex cycles of all boundary components."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if not len(boundary_edges):
        return []
    adjacency: dict[int, List[int]] = {}
    unused: set[tuple[int, int]] = set()
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
                edge = (min(current, candidate), max(current, candidate))
                if candidate != previous and edge in unused:
                    unused.remove(edge)
                    next_vertex = candidate
                    break
            if next_vertex is None:
                raise ValueError("The mesh boundary is open or non-manifold.")
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def mesh_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def _shortest_path_to_other_boundary(
    mesh: trimesh.Trimesh, source_loop: np.ndarray,
    other_loops: Sequence[np.ndarray],
) -> np.ndarray:
    """Find the shortest mesh-edge path from one boundary to another."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = mesh_edges_from_faces(np.asarray(mesh.faces, dtype=np.int64))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    source = set(map(int, source_loop))
    terminals = set(map(int, np.concatenate(other_loops)))
    adjacency: dict[int, List[tuple[int, float]]] = {}
    for (a_raw, b_raw), length in zip(edges, lengths):
        a, b = int(a_raw), int(b_raw)
        if a not in terminals or a in source:
            adjacency.setdefault(a, []).append((b, float(length)))
        if b not in terminals or b in source:
            adjacency.setdefault(b, []).append((a, float(length)))
    super_source = len(vertices)
    positive = lengths[lengths > 1e-12]
    connector = max(float(np.median(positive)) if len(positive) else 1.0, 1.0) * 1e-9
    for vertex in source:
        adjacency.setdefault(super_source, []).append((vertex, connector))
    distance = {super_source: 0.0}
    predecessor: dict[int, int] = {}
    queue = [(0.0, super_source)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance.get(current):
            continue
        for neighbour, weight in adjacency.get(current, ()):
            candidate = current_distance + weight
            if candidate < distance.get(neighbour, float("inf")):
                distance[neighbour] = candidate
                predecessor[neighbour] = current
                heapq.heappush(queue, (candidate, neighbour))
    reachable = [vertex for vertex in terminals if vertex in distance]
    if not reachable:
        raise ValueError("Could not connect the mesh boundaries by an interior seam.")
    target = min(reachable, key=lambda vertex: distance[vertex])
    path = [int(target)]
    while path[-1] != super_source:
        if path[-1] not in predecessor:
            raise ValueError("The seam predecessor chain is incomplete.")
        path.append(predecessor[path[-1]])
    return np.asarray(path[-2::-1], dtype=np.int64)


def _cut_along_path(mesh: trimesh.Trimesh, seam: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Duplicate the seam vertices on one side, producing a disk-like mesh."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    seam = np.asarray(seam, dtype=np.int64)
    if len(seam) < 2 or len(np.unique(seam)) != len(seam):
        raise ValueError("The seam must be a simple path with at least two vertices.")
    seam_edges = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(seam[:-1], seam[1:])}
    incident: dict[int, List[int]] = {int(v): [] for v in seam}
    edge_faces: dict[tuple[int, int], List[int]] = {}
    for fi, tri_raw in enumerate(faces):
        tri = [int(v) for v in tri_raw]
        for v in tri:
            if v in incident: incident[v].append(fi)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (min(a, b), max(a, b))
            if edge in seam_edges: edge_faces.setdefault(edge, []).append(fi)
    duplicate_faces: dict[int, set[int]] = {}
    for vertex, incident_faces in incident.items():
        adjacency = {fi: set() for fi in incident_faces}
        for fi in incident_faces:
            for other_raw in faces[fi]:
                other = int(other_raw)
                edge = (min(vertex, other), max(vertex, other))
                if other != vertex and edge not in seam_edges:
                    for fj in incident_faces:
                        if fj != fi and other in faces[fj]: adjacency[fi].add(fj)
        remaining = set(incident_faces); components = []
        while remaining:
            stack = [remaining.pop()]; component = set(stack)
            while stack:
                neighbours = adjacency[stack.pop()] & remaining
                remaining.difference_update(neighbours); stack.extend(neighbours); component.update(neighbours)
            components.append(component)
        if len(components) != 2:
            raise ValueError(f"Seam vertex {vertex} is not locally cuttable.")
        duplicate_faces[vertex] = components[1]
    duplicate_map = {int(v): len(vertices) + i for i, v in enumerate(seam)}
    cut_faces = faces.copy()
    for fi, tri in enumerate(cut_faces):
        for li, vertex_raw in enumerate(tri):
            if int(vertex_raw) in duplicate_faces and fi in duplicate_faces[int(vertex_raw)]:
                tri[li] = duplicate_map[int(vertex_raw)]
    cut_mesh = trimesh.Trimesh(vertices=np.vstack((vertices, vertices[seam])), faces=cut_faces, process=False)
    seam_vertices = np.concatenate((seam, np.arange(len(vertices), len(vertices) + len(seam), dtype=np.int64)))
    return cut_mesh, seam_vertices


def _select_mesh_file() -> str:
    import tkinter as tk
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


def _precut_two_boundaries(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Cut a two-boundary mesh along its globally shortest edge geodesic."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = ordered_boundary_loops(faces)
    if len(loops) != 2:
        return mesh, np.empty((0, 2, 3), dtype=float)

    seam = _shortest_path_to_other_boundary(
        mesh, loops[0], [loops[1]]
    )
    vertices = np.asarray(mesh.vertices, dtype=float)
    seam_edges = vertices[np.column_stack([seam[:-1], seam[1:]])]
    seam_length = float(np.sum(np.linalg.norm(
        vertices[seam[1:]] - vertices[seam[:-1]], axis=1
    )))
    cut_mesh, _ = _cut_along_path(mesh, seam)
    after = len(ordered_boundary_loops(
        np.asarray(cut_mesh.faces, dtype=np.int64)
    ))
    if after != 1:
        raise ValueError(
            "The shortest boundary geodesic did not produce a disk-like "
            f"OptCuts input (expected 1 boundary, found {after})."
        )
    print(
        "Two-boundary mesh: pre-cut along the globally shortest boundary "
        f"geodesic ({len(seam)} vertices, length {seam_length:.6g})."
    )
    return cut_mesh, seam_edges


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
        "10",                  # 官方离线可视化模式：每次几何/拓扑交替后刷新
        input_obj.relative_to(OPTCUTS_ROOT).as_posix(),
        "0.999",               # 官方 README 的初始 lambda
        "1",
        "0",                   # methodType 0: OptCuts
        "4.1",                 # 官方 README 示例的 SD 畸变上界
        "1",                   # enforce bijectivity
        "0",                   # random one-point initial cut
        "python",
    ]
    completed = subprocess.run(command, cwd=OPTCUTS_ROOT, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Official OptCuts exited with code {completed.returncode}.")

    # Every prepared input has a unique UUID stem.  Official OptCuts names its
    # result directory from that stem, for example
    # ``optcuts_<uuid>_Tutte_...``.  Only accept that run's output: falling
    # back to any historical finalResult_mesh.obj displays stale content after
    # an interrupted or failed run.
    expected_prefix = input_obj.stem + "_"
    candidates = [
        result
        for result in output_root.glob("**/finalResult_mesh.obj")
        if result.parent.name.startswith(expected_prefix)
    ]
    if not candidates:
        raise FileNotFoundError(
            "This OptCuts run did not produce finalResult_mesh.obj; "
            "the run may have been stopped before completion. Historical "
            "results will not be displayed."
        )
    result = max(candidates, key=lambda item: item.stat().st_mtime)
    print(f"Current OptCuts result: {result}")
    return result


def _read_obj_uv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    texture_coordinates = []
    texture_faces = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("vt "):
                values = line.split()
                texture_coordinates.append((float(values[1]), float(values[2])))
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    continue
                indices = []
                for corner in corners:
                    parts = corner.split("/")
                    if len(parts) < 2 or not parts[1]:
                        raise ValueError("OptCuts result contains no texture-coordinate indices.")
                    indices.append(int(parts[1]) - 1)
                texture_faces.append(indices)
    uv = np.asarray(texture_coordinates, dtype=float)
    faces = np.asarray(texture_faces, dtype=np.int64)
    if not len(uv) or not len(faces):
        raise ValueError("Cannot read UV coordinates from the OptCuts result.")
    return uv, faces


def _read_obj_cut_edges(path: Path, uv_tolerance: float = 1.0e-8) -> np.ndarray:
    """Extract geometric seam edges from UV discontinuities in an OptCuts OBJ.

    OptCuts does not export a separate ``seam_edges`` section.  A cut is
    represented by two adjacent triangles sharing the same geometric edge but
    using different texture-coordinate corners at one or both endpoints.
    Boundary edges with only one incident face are deliberately ignored.
    """
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

    # Each geometric edge stores the first face's UV coordinates at its two
    # endpoints. A second incident face with a discontinuous UV edge is a cut.
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


def _play_iteration_animation_matplotlib_legacy(
    mesh: trimesh.Trimesh, result_obj: Path
) -> None:
    """Render the evolving UV checker grid directly on the 3D workpiece."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    gif_path = result_obj.with_name("anim.gif")
    mesh_frame_directory = result_obj.with_name("mesh_frames")
    if not gif_path.exists():
        print(f"OptCuts iteration animation was not found: {gif_path}")
        return

    def read_gif(path: Path):
        images = []
        frame_durations = []
        with Image.open(path) as animation:
            frame_count = getattr(animation, "n_frames", 1)
            for frame_index in range(frame_count):
                animation.seek(frame_index)
                images.append(np.asarray(animation.convert("RGB")).copy())
                frame_durations.append(max(int(animation.info.get("duration", 100)), 20))
        return images, frame_durations

    frames, durations = read_gif(gif_path)
    if not frames:
        return
    mesh_frame_paths = sorted(
        mesh_frame_directory.glob("frame_*.obj"),
        key=lambda item: int(item.stem.split("_")[-1]),
    )
    if not mesh_frame_paths:
        print(f"OptCuts 3D parameter-grid frames were not found: {mesh_frame_directory}")
        return
    synchronized_count = min(len(frames), len(mesh_frame_paths))
    frames = frames[:synchronized_count]
    mesh_frame_paths = mesh_frame_paths[:synchronized_count]

    figure = plt.figure(figsize=(16, 7))
    axes_3d = figure.add_subplot(121, projection="3d")
    first_vertices, first_faces, first_uv, first_uv_faces = _read_iteration_obj(mesh_frame_paths[0])
    uv_center = first_uv[first_uv_faces].mean(axis=1)
    texture_scale = 10.0 / max(float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))), 1.0e-12)
    checker = (np.floor(uv_center[:, 0] * texture_scale * 2.0)
               + np.floor(uv_center[:, 1] * texture_scale * 2.0)).astype(int) & 1
    checker_colors = np.where(checker[:, None] == 0,
                              np.array([[0.08, 0.08, 0.08, 1.0]]),
                              np.array([[0.92, 0.92, 0.92, 1.0]]))
    surface_artist = Poly3DCollection(
        first_vertices[first_faces], facecolors=checker_colors,
        edgecolors=(0.25, 0.25, 0.25, 0.35), linewidth=0.15,
    )
    axes_3d.add_collection3d(surface_artist)
    original_vertices = np.asarray(mesh.vertices, dtype=float)
    minimum, maximum = original_vertices.min(axis=0), original_vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * max(float(np.max(maximum - minimum)), np.finfo(float).eps)
    axes_3d.set_xlim(center[0] - radius, center[0] + radius)
    axes_3d.set_ylim(center[1] - radius, center[1] + radius)
    axes_3d.set_zlim(center[2] - radius, center[2] + radius)
    axes_3d.set_box_aspect((1, 1, 1))
    axes_3d.set_xlabel("x")
    axes_3d.set_ylabel("y")
    axes_3d.set_zlabel("z")
    axes_3d.set_title("Parameter grid mapped on 3D workpiece")
    axes_uv = figure.add_subplot(122)
    image_artist = axes_uv.imshow(frames[0])
    title = axes_uv.set_title(f"OptCuts iteration 1/{len(frames)}")
    axes_uv.axis("off")

    def update(frame_index: int):
        current_vertices, current_faces, current_uv, current_uv_faces = _read_iteration_obj(
            mesh_frame_paths[frame_index]
        )
        current_uv_center = current_uv[current_uv_faces].mean(axis=1)
        current_checker = (
            np.floor(current_uv_center[:, 0] * texture_scale * 2.0)
            + np.floor(current_uv_center[:, 1] * texture_scale * 2.0)
        ).astype(int) & 1
        current_colors = np.where(current_checker[:, None] == 0,
                                  np.array([[0.08, 0.08, 0.08, 1.0]]),
                                  np.array([[0.92, 0.92, 0.92, 1.0]]))
        surface_artist.set_verts(current_vertices[current_faces])
        surface_artist.set_facecolors(current_colors)
        image_artist.set_data(frames[frame_index])
        title.set_text(f"OptCuts iteration {frame_index + 1}/{len(frames)} (looping)")
        return surface_artist, image_artist, title

    # Matplotlib 只接受统一间隔；采用官方 GIF 各帧时长的中位数并永久循环。
    interval = int(np.median(durations))
    final_repeat_count = max(1, int(round(FINAL_FRAME_HOLD_MS / max(interval, 1))))
    playback_frames = list(range(len(frames))) + [len(frames) - 1] * final_repeat_count
    player = FuncAnimation(
        figure,
        update,
        frames=playback_frames,
        interval=interval,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    figure._optcuts_animation = player
    figure.tight_layout()
    plt.show()


def _show_result_matplotlib_legacy(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    cut_edges: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    figure = plt.figure(figsize=(15, 7))
    axes_3d = figure.add_subplot(121, projection="3d")
    axes_3d.add_collection3d(Poly3DCollection(
        vertices[faces], facecolor=(0.68, 0.78, 0.88, 0.65),
        edgecolor=(0.2, 0.2, 0.2, 0.35), linewidth=0.25,
    ))
    if cut_edges is not None and len(cut_edges):
        for segment in cut_edges:
            axes_3d.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color="crimson", linewidth=2.0, alpha=0.95,
            )
        axes_3d.plot([], [], [], color="crimson", linewidth=2.0,
                     label=f"OptCuts seams ({len(cut_edges)})")
        axes_3d.legend(loc="best")
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * max(float(np.max(maximum - minimum)), np.finfo(float).eps)
    axes_3d.set_xlim(center[0] - radius, center[0] + radius)
    axes_3d.set_ylim(center[1] - radius, center[1] + radius)
    axes_3d.set_zlim(center[2] - radius, center[2] + radius)
    axes_3d.set_box_aspect((1, 1, 1))
    axes_3d.set_xlabel("x")
    axes_3d.set_ylabel("y")
    axes_3d.set_zlabel("z")
    axes_3d.set_title(
        f"Original 3D mesh + OptCuts seams ({len(cut_edges) if cut_edges is not None else 0})"
    )

    axes_uv = figure.add_subplot(122)
    axes_uv.triplot(uv[:, 0], uv[:, 1], uv_faces, color="black", linewidth=0.45)
    axes_uv.set_aspect("equal", adjustable="box")
    axes_uv.set_xlabel("u")
    axes_uv.set_ylabel("v")
    axes_uv.set_title("OptCuts UV mesh")
    figure.tight_layout()
    plt.show()


# VTK display implementations. The optimization and seam computation stay intact.
def _vtk_triangle_mesh(points: np.ndarray, faces: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))
    data = vtk.vtkPolyData()
    data.SetPoints(vtk_points)
    data.SetPolys(vtk_faces)
    return data


def _vtk_polyline(points: np.ndarray, closed: bool = False):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = np.asarray(points, dtype=float)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    indices = np.arange(len(points), dtype=np.int64)
    if closed and len(indices):
        indices = np.r_[indices, indices[0]]
    records = np.r_[len(indices), indices].astype(np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(1, numpy_to_vtkIdTypeArray(records, deep=True))
    data = vtk.vtkPolyData()
    data.SetPoints(vtk_points)
    data.SetLines(cells)
    return data


def _vtk_add_title(renderer, text: str, color=(1.0, 1.0, 1.0)):
    import vtk

    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetTextProperty().SetFontSize(21)
    actor.GetTextProperty().SetColor(*color)
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.025, 0.945)
    renderer.AddActor2D(actor)
    return actor


def _play_iteration_animation(mesh: trimesh.Trimesh, result_obj: Path) -> None:
    """Play synchronized 3D and UV iteration meshes with VTK."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    frame_directory = result_obj.with_name("mesh_frames")
    frame_paths = sorted(
        frame_directory.glob("frame_*.obj"),
        key=lambda item: int(item.stem.split("_")[-1]),
    )
    if not frame_paths:
        print(f"OptCuts VTK iteration frames were not found: {frame_directory}")
        return

    model_extent = max(
        float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))), 1.0e-12
    )
    texture_scale = 10.0 / model_extent

    def build_frame(frame_index: int):
        vertices, faces, uv, uv_faces = _read_iteration_obj(frame_paths[frame_index])
        model_data = _vtk_triangle_mesh(vertices, faces)
        uv_centers = uv[uv_faces].mean(axis=1)
        checker = (
            np.floor(uv_centers[:, 0] * texture_scale * 2.0)
            + np.floor(uv_centers[:, 1] * texture_scale * 2.0)
        ).astype(np.int64) & 1
        colors = np.where(
            checker[:, None] == 0,
            np.array([[20, 20, 20, 255]], dtype=np.uint8),
            np.array([[235, 235, 235, 255]], dtype=np.uint8),
        )
        color_array = numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        color_array.SetNumberOfComponents(4)
        model_data.GetCellData().SetScalars(color_array)
        uv_points = np.column_stack([uv, np.zeros(len(uv), dtype=float)])
        return model_data, _vtk_triangle_mesh(uv_points, uv_faces)

    model_data, uv_data = build_frame(0)
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputData(model_data)
    model_mapper.SetScalarModeToUseCellData()
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().EdgeVisibilityOn()
    model_actor.GetProperty().SetEdgeColor(0.18, 0.18, 0.18)
    model_actor.GetProperty().SetLineWidth(0.5)

    uv_mapper = vtk.vtkPolyDataMapper()
    uv_mapper.SetInputData(uv_data)
    uv_mapper.ScalarVisibilityOff()
    uv_actor = vtk.vtkActor()
    uv_actor.SetMapper(uv_mapper)
    uv_actor.GetProperty().SetColor(0.72, 0.84, 0.95)
    uv_actor.GetProperty().EdgeVisibilityOn()
    uv_actor.GetProperty().SetEdgeColor(0.06, 0.06, 0.06)

    model_renderer = vtk.vtkRenderer()
    model_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
    model_renderer.SetBackground(0.12, 0.15, 0.19)
    model_renderer.AddActor(model_actor)
    model_renderer.ResetCamera()
    uv_renderer = vtk.vtkRenderer()
    uv_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
    uv_renderer.SetBackground(0.96, 0.96, 0.96)
    uv_renderer.AddActor(uv_actor)
    uv_renderer.ResetCamera()
    _vtk_add_title(model_renderer, "Parameter grid on 3D workpiece")
    uv_title = _vtk_add_title(
        uv_renderer, f"OptCuts iteration 1/{len(frame_paths)}", (0.08, 0.08, 0.08)
    )

    window = vtk.vtkRenderWindow()
    window.SetWindowName("OptCuts Iterations - VTK")
    window.SetSize(1400, 760)
    window.AddRenderer(model_renderer)
    window.AddRenderer(uv_renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    interval = 120
    final_repeats = max(1, int(round(FINAL_FRAME_HOLD_MS / interval)))
    playback = list(range(len(frame_paths))) + [len(frame_paths) - 1] * final_repeats
    state = {"position": 0}

    def advance(_caller, _event):
        state["position"] = (state["position"] + 1) % len(playback)
        frame_index = playback[state["position"]]
        current_model, current_uv = build_frame(frame_index)
        model_mapper.SetInputData(current_model)
        uv_mapper.SetInputData(current_uv)
        uv_title.SetInput(
            f"OptCuts iteration {frame_index + 1}/{len(frame_paths)} (looping)"
        )
        window.Render()

    interactor.AddObserver("TimerEvent", advance)
    interactor.Initialize()
    window.Render()
    interactor.CreateRepeatingTimer(interval)
    interactor.Start()


def _show_result(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    cut_edges: np.ndarray | None = None,
    uv_decomposition: dict[str, object] | None = None,
) -> None:
    """Show final 3D seams and the UV mesh in a VTK split view."""
    import vtk

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputData(_vtk_triangle_mesh(vertices, faces))
    model_mapper.ScalarVisibilityOff()
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.GetProperty().SetColor(0.68, 0.78, 0.88)
    model_actor.GetProperty().SetOpacity(0.78)
    model_actor.GetProperty().EdgeVisibilityOn()
    model_actor.GetProperty().SetEdgeColor(0.18, 0.18, 0.18)

    model_renderer = vtk.vtkRenderer()
    model_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
    model_renderer.SetBackground(0.12, 0.15, 0.19)
    model_renderer.AddActor(model_actor)
    seam_count = len(cut_edges) if cut_edges is not None else 0
    if seam_count:
        seam_points = vtk.vtkPoints()
        seam_lines = vtk.vtkCellArray()
        for segment in cut_edges:
            first = seam_points.InsertNextPoint(*segment[0])
            second = seam_points.InsertNextPoint(*segment[1])
            seam_lines.InsertNextCell(2)
            seam_lines.InsertCellPoint(first)
            seam_lines.InsertCellPoint(second)
        seam_data = vtk.vtkPolyData()
        seam_data.SetPoints(seam_points)
        seam_data.SetLines(seam_lines)
        seam_mapper = vtk.vtkPolyDataMapper()
        seam_mapper.SetInputData(seam_data)
        seam_actor = vtk.vtkActor()
        seam_actor.SetMapper(seam_mapper)
        seam_actor.GetProperty().SetColor(0.86, 0.08, 0.18)
        seam_actor.GetProperty().SetLineWidth(4.0)
        model_renderer.AddActor(seam_actor)
    model_renderer.ResetCamera()
    _vtk_add_title(model_renderer, f"Original 3D mesh + OptCuts seams ({seam_count})")

    uv_points = np.column_stack([uv, np.zeros(len(uv), dtype=float)])
    uv_mapper = vtk.vtkPolyDataMapper()
    uv_mapper.SetInputData(_vtk_triangle_mesh(uv_points, uv_faces))
    uv_mapper.ScalarVisibilityOff()
    uv_actor = vtk.vtkActor()
    uv_actor.SetMapper(uv_mapper)
    uv_actor.GetProperty().SetColor(0.72, 0.84, 0.95)
    uv_actor.GetProperty().EdgeVisibilityOn()
    uv_actor.GetProperty().SetEdgeColor(0.05, 0.05, 0.05)
    uv_actor.GetProperty().SetLineWidth(0.7)
    uv_renderer = vtk.vtkRenderer()
    uv_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
    uv_renderer.SetBackground(0.96, 0.96, 0.96)
    uv_renderer.AddActor(uv_actor)
    if uv_decomposition is not None:
        # Overlay the retained rectangle and the non-overlapping remainder
        # triangles so the partition is inspectable directly in UV space.
        rectangle = np.asarray(uv_decomposition["rectangle"], dtype=float)
        rectangle_points = np.column_stack((rectangle, np.full(len(rectangle), 0.01)))
        rectangle_data = _vtk_polyline(rectangle_points, closed=True)
        rectangle_actor = vtk.vtkActor()
        rectangle_mapper = vtk.vtkPolyDataMapper()
        rectangle_mapper.SetInputData(rectangle_data)
        rectangle_actor.SetMapper(rectangle_mapper)
        rectangle_actor.GetProperty().SetColor(0.88, 0.12, 0.10)
        rectangle_actor.GetProperty().SetLineWidth(4.0)
        uv_renderer.AddActor(rectangle_actor)

        remainder_triangles = uv_decomposition.get("triangles", [])
        if remainder_triangles:
            triangle_points = np.vstack([
                np.column_stack((triangle, np.full(3, 0.008)))
                for triangle in remainder_triangles
            ])
            triangle_faces = np.arange(len(triangle_points), dtype=np.int64).reshape(-1, 3)
            remainder_mapper = vtk.vtkPolyDataMapper()
            remainder_mapper.SetInputData(_vtk_triangle_mesh(triangle_points, triangle_faces))
            remainder_actor = vtk.vtkActor()
            remainder_actor.SetMapper(remainder_mapper)
            remainder_actor.GetProperty().SetRepresentationToWireframe()
            remainder_actor.GetProperty().SetColor(0.10, 0.55, 0.20)
            remainder_actor.GetProperty().SetLineWidth(1.4)
            uv_renderer.AddActor(remainder_actor)
    uv_renderer.ResetCamera()
    title = "OptCuts UV mesh"
    if uv_decomposition is not None:
        title += " | red: max rectangle, green: remainder triangles"
    _vtk_add_title(uv_renderer, title, (0.08, 0.08, 0.08))

    window = vtk.vtkRenderWindow()
    window.SetWindowName("OptCuts Final Result - VTK")
    window.SetSize(1400, 760)
    window.AddRenderer(model_renderer)
    window.AddRenderer(uv_renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    interactor.Initialize()
    window.Render()
    interactor.Start()


def main() -> None:
    selected = _select_mesh_file()
    if not selected:
        return
    mesh = _load_mesh(selected)
    optcuts_mesh, shortest_boundary_seam = _precut_two_boundaries(mesh)
    official_input = _prepare_official_input(optcuts_mesh)
    result_obj = _run_official_optcuts(official_input)
    uv, uv_faces = _read_obj_uv(result_obj)
    uv_decomposition = decompose_uv_domain(uv, uv_faces)
    optimized_cut_edges = _read_obj_cut_edges(result_obj)
    if len(shortest_boundary_seam) and len(optimized_cut_edges):
        cut_edges = np.vstack([shortest_boundary_seam, optimized_cut_edges])
    elif len(shortest_boundary_seam):
        cut_edges = shortest_boundary_seam
    else:
        cut_edges = optimized_cut_edges
    print(f"Pre-cut shortest-boundary seam edges: {len(shortest_boundary_seam)}")
    print(f"Additional detected OptCuts seam edges: {len(optimized_cut_edges)}")
    print(
        "UV maximum rectangle bounds: "
        f"{uv_decomposition['rectangle_bounds']}; "
        f"remainder triangles: {len(uv_decomposition['triangles'])}"
    )
    _play_iteration_animation(optcuts_mesh, result_obj)
    _show_result(mesh, uv, uv_faces, cut_edges, uv_decomposition)


if __name__ == "__main__":
    main()
#预先切割
