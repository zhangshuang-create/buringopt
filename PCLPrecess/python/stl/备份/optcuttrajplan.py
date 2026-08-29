"""Run OptCuts and plan a seam-aware coverage trajectory on its UV result."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh

import LSCM as lscm_preprocess
import loft_uv_parameterization as loft_preprocess
# OptCut.py lives in ``PCLPrecess/python/stl``.  The OptCuts checkout is a
# sibling of ``python`` under ``PCLPrecess``, so parents[2] is the project
# root (parents[1] would incorrectly point at ``PCLPrecess/python``).
ROOT = Path(__file__).resolve().parents[2]
OPTCUTS_ROOT = ROOT / "third_party" / "OptCuts"
OPTCUTS_EXE = OPTCUTS_ROOT / "build" / "Release" / "OptCuts_bin.exe"
FINAL_FRAME_HOLD_MS = 2000
DEFAULT_TRAJECTORY_SPACING = 30.0
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
) -> tuple[trimesh.Trimesh, PrecutSeam | None]:
    """Cut a two-boundary mesh along loft's shorter three-point-plane curve."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = lscm_preprocess.ordered_boundary_loops(faces)
    if len(loops) != 2:
        return mesh, None

    seam, seam_metrics = _loft_three_point_seam(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float)
    # Burner files use +Z from the lower opening to the upper opening.  This
    # stable ordering also fixes the sign of the local planning y axis.
    if vertices[int(seam[0]), 2] > vertices[int(seam[-1]), 2]:
        seam = seam[::-1].copy()
    seam_length = float(np.sum(np.linalg.norm(
        vertices[seam[1:]] - vertices[seam[:-1]], axis=1
    )))
    original_vertex_count = len(vertices)
    cut_mesh, _ = lscm_preprocess._cut_along_path(mesh, seam)
    seam_data = PrecutSeam(
        original_vertex_ids=np.asarray(seam, dtype=np.int64),
        cut_vertex_pairs=np.column_stack([
            seam,
            np.arange(original_vertex_count, original_vertex_count + len(seam), dtype=np.int64),
        ]),
        xyz=vertices[seam].copy(),
    )
    after = len(lscm_preprocess.ordered_boundary_loops(
        np.asarray(cut_mesh.faces, dtype=np.int64)
    ))
    if after != 1:
        raise ValueError(
            "The loft three-point-plane seam did not produce a disk-like "
            f"OptCuts input (expected 1 boundary, found {after})."
        )
    print(
        "Two-boundary mesh: loft three-point plane generated two main "
        f"intersections ({seam_metrics['intersection_length_1']:.6g}, "
        f"{seam_metrics['intersection_length_2']:.6g}); selected the shorter "
        f"one ({seam_metrics['selected_intersection_length']:.6g}). Snapped "
        f"edge seam: {len(seam)} vertices, length {seam_length:.6g}, maximum "
        f"deviation {seam_metrics['maximum_snap_deviation']:.6g}."
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


def _snap_intersection_curve_to_mesh_edges(
    mesh: trimesh.Trimesh,
    curve: np.ndarray,
    bottom_vertices: np.ndarray,
    top_vertices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find an edge path strongly constrained to one selected plane intersection."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = lscm_preprocess.mesh_edges_from_faces(faces)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    positive_lengths = edge_lengths[edge_lengths > EPS]
    typical_edge = float(np.median(positive_lengths)) if len(positive_lengths) else 1.0
    dense_curve = _dense_polyline_samples(curve, max(0.5 * typical_edge, EPS))
    curve_tree = cKDTree(dense_curve)
    distance_to_curve, _ = curve_tree.query(vertices)

    start = int(bottom_vertices[np.argmin(np.linalg.norm(
        vertices[bottom_vertices] - curve[0], axis=1
    ))])
    end = int(top_vertices[np.argmin(np.linalg.norm(
        vertices[top_vertices] - curve[-1], axis=1
    ))])
    boundary_set = set(map(int, np.concatenate([bottom_vertices, top_vertices])))
    guidance_scale = max(2.0 * typical_edge, EPS)
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for (a_raw, b_raw), length in zip(edges, edge_lengths):
        a, b = int(a_raw), int(b_raw)
        deviation = 0.5 * (distance_to_curve[a] + distance_to_curve[b]) / guidance_scale
        guided_weight = float(length) * (1.0 + 50.0 * deviation * deviation)
        # The seam may start/end on an opening but must not run along either
        # opening boundary before entering the surface interior.
        if b not in boundary_set or b == end:
            rows.append(a)
            columns.append(b)
            weights.append(guided_weight)
        if a not in boundary_set or a == end:
            rows.append(b)
            columns.append(a)
            weights.append(guided_weight)
    graph = coo_matrix((weights, (rows, columns)), shape=(len(vertices), len(vertices))).tocsr()
    distance, predecessors = dijkstra(
        graph, directed=True, indices=start, return_predecessors=True
    )
    if not np.isfinite(distance[end]):
        raise ValueError("Could not snap the selected loft intersection to mesh edges.")
    path = [end]
    while path[-1] != start:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("The loft-guided seam predecessor chain is incomplete.")
        path.append(predecessor)
    path = np.asarray(path[::-1], dtype=np.int64)
    maximum_deviation = float(np.max(distance_to_curve[path]))
    return path, maximum_deviation


def _loft_three_point_seam(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict[str, float]]:
    """Build loft's three-point plane and select its shorter main intersection."""
    boundary_loops = loft_preprocess.extract_boundary_loops(mesh)
    top, bottom = loft_preprocess.classify_top_bottom_boundaries(boundary_loops)
    _, pca_axes, _ = loft_preprocess.compute_pca(top.points)
    pca_second_direction = pca_axes[1]
    try:
        third_point = loft_preprocess._upper_boundary_contact_point(top)
    except Exception:
        third_point = loft_preprocess._upper_boundary_pca_second_point(
            top, pca_second_direction
        )
    plane = loft_preprocess.build_reference_plane(
        top.center, bottom.center, third_point
    )
    curves = [
        np.asarray(curve, dtype=float)
        for curve in loft_preprocess.intersect_mesh_with_plane(mesh, plane)
        if len(curve) >= 2 and _polyline_length(curve) > EPS
    ]
    if len(curves) < 2:
        raise ValueError(
            "The loft three-point plane produced fewer than two usable intersections."
        )
    # loft_uv_parameterization treats the two longest components as the main
    # side intersections. Select the shorter one as the physical cut curve.
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
    seam, maximum_deviation = _snap_intersection_curve_to_mesh_edges(
        mesh, selected, bottom_ids, top_ids
    )
    return seam, {
        "intersection_length_1": curve_lengths[0],
        "intersection_length_2": curve_lengths[1],
        "selected_intersection_length": curve_lengths[selected_index],
        "maximum_snap_deviation": maximum_deviation,
    }


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


def _read_obj_uv_data(path: Path) -> ObjUVData:
    """Read the geometric and texture corner indices needed for UV -> XYZ."""
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
    """Resolve each saved seam edge without assuming continuity at cut junctions."""
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
        # A pre-cut side is a geometric boundary edge and normally has one
        # incident face.  If duplicate records exist, prefer the longest UV
        # representation rather than a collapsed/degenerate one.
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
        # The two UV edges use the same physical interpolation parameter.  Find
        # the continuous minimum of |B(t) - A(t)| on this edge pair instead of
        # checking only its two ends and midpoint; otherwise reference line 1
        # can miss the true equivalent seam point and the waist transition is
        # attached to a neighbouring scan.
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
    """Intersect x=constant with the actual triangle union, never a hull."""
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
    # One offset line remains one complete line even when an OptCuts chart gap
    # splits its triangle intersections into several intervals.
    return min(item[0] for item in candidates), max(item[1] for item in candidates)


def optimize_spacing(nominal: float, dimensions: np.ndarray) -> Optimum:
    """Optimize one shared spacing for center, upper, and lower regions."""
    nominal = float(nominal)
    if nominal <= 0.0:
        raise ValueError("Nominal trajectory spacing must be greater than zero.")
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Spacing dimensions must contain center, upper, and lower widths.")
    if dimensions[0] <= 0.0 or np.any(dimensions[1:] < 0.0):
        raise ValueError("Center width must be positive and transition widths cannot be negative.")

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
            np.rint(dimensions[None, 1:] / candidates[:, None]).astype(int),
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
    """Project to one saved seam copy and return its exact physical counterpart."""
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

    # Opening overtravel must stay on one side of the opening.  Using
    # previous+exit and following-entry independently can place the two control
    # points on opposite sides when the local scan tangents are noisy or not
    # exactly antiparallel; that creates the visible diagonal connector crossing
    # unrelated tracks.  The common outside direction keeps the connector as a
    # true boundary-side return motion.
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
        note=note,
    )


def _trim_mapped_segment(
    segment: TrajectorySegment,
    distance: float,
    from_start: bool,
) -> TrajectorySegment:
    """Trim a mapped trajectory by 3D arclength while interpolating its UV."""
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
    xyz = np.asarray(segment.xyz, dtype=float)
    uv = np.asarray(segment.uv, dtype=float)
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
    """Generate a 3D coordinated-motion blend with exact endpoint tangents."""
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
    """Replace mapped seam corners with 3D-only G1 coordinated paths.

    UV planning has already finished when this function runs. The SEAM_JUMP
    is retained as a UV/topology marker but carries no executable XYZ points.
    """
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
            if lead <= EPS:
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
) -> TrajectorySegment:
    uv = _sample_scan_uv(x_value, interval, upward, sample_step, frame)
    return TrajectorySegment(
        "SURFACE_SCAN", uv,
        mapper.map(uv, snap_tolerance, bridge_outside=True), True, note,
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
    """Build a scan starting exactly at a seam destination, using that point's x as the scan x.

    This is the seam-driven counterpart to _make_surface_scan: instead of using
    a pre-spaced grid x and then snapping the start point, we derive the x from
    the seam destination itself, ensuring no kink at the join.

    Returns None if no valid scan interval exists at the destination x.
    """
    dest_local = _to_local(np.asarray(destination_uv, dtype=float)[None, :], frame)[0]
    x_val = float(dest_local[0])
    dest_y = float(dest_local[1])

    # Span every interval belonging to this region. A chart gap must not split
    # or delete one physical offset line.
    all_intervals = _line_domain_intervals(local_uv, uv_faces, x_val, tolerance)
    if not all_intervals:
        return None  # No mesh at this x position
    interval = _region_interval(local_uv, uv_faces, x_val, region, tolerance)
    if interval is None:
        interval = (
            min(item[0] for item in all_intervals),
            max(item[1] for item in all_intervals),
        )

    low, high = interval
    # Clip so the scan starts from the seam destination, not the full interval boundary
    if upward:
        low = float(np.clip(dest_y, low, high))
    else:
        high = float(np.clip(dest_y, low, high))

    # Skip zero-length scans (destination is already at boundary)
    if abs(high - low) < tolerance:
        return None

    uv = _sample_scan_uv(x_val, (low, high), upward, sample_step, frame)
    # Hard-force first point to exact seam destination (eliminates floating-point drift)
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
    """Keep every intersecting center offset and bridge all chart gaps."""
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


def generate_seam_aware_trajectory(
    data: ObjUVData,
    seam: PrecutSeam,
    target_spacing: float,
    sample_step: float | None = None,
    overtravel_factor: float = 2.0,
) -> tuple[list[TrajectorySegment], PlanningFrame, dict[str, object]]:
    """Plan lower seam pairs, a central snake, then upper seam pairs."""
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

    dimensions = np.asarray([
        frame.waist_width,
        0.5 * (extents["upper_left"] + extents["upper_right"]),
        0.5 * (extents["lower_left"] + extents["lower_right"]),
    ], dtype=float)
    optimum = optimize_spacing(nominal_spacing, dimensions)
    target_spacing = optimum.spacing
    print(
        "Spacing optimization: "
        f"dimensions(center, upper, lower)={dimensions.tolist()}, "
        f"nominal={nominal_spacing:.6g}, optimized={target_spacing:.6g}, "
        f"counts={optimum.counts.tolist()}, "
        f"actual={optimum.actual_spacings.tolist()}."
    )
    sample_step = target_spacing / 4.0 if sample_step is None else float(sample_step)
    if sample_step <= 0.0:
        raise ValueError("Trajectory sampling step must be greater than zero.")
    boundary_snap_tolerance = max(100.0 * (1.0e-9 * uv_scale), 0.5 * sample_step)
    lower_count = int(optimum.counts[2])
    upper_count = int(optimum.counts[1])

    lower_left_x, lower_left_spacing = _balanced_outer_positions(
        extents["lower_left"], lower_count, 0.0, -1.0
    )
    lower_right_x, lower_right_spacing = _balanced_outer_positions(
        extents["lower_right"], lower_count, frame.waist_width, 1.0
    )
    upper_left_x, upper_left_spacing = _balanced_outer_positions(
        extents["upper_left"], upper_count, 0.0, -1.0
    )
    upper_right_x, upper_right_spacing = _balanced_outer_positions(
        extents["upper_right"], upper_count, frame.waist_width, 1.0
    )
    center_x, center_spacing = _balanced_center_positions(
        frame.waist_width, int(optimum.counts[0])
    )

    # Lower tracks run from the opening towards the waist, outermost first.
    lower_left = _available_tracks(lower_left_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    lower_right = _available_tracks(lower_right_x[::-1], "lower", local_uv, data.uv_faces, tolerance)
    # Upper tracks leave the waist and finish d/2 from the outer top corners.
    upper_left = _available_tracks(upper_left_x, "upper", local_uv, data.uv_faces, tolerance)
    upper_right = _available_tracks(upper_right_x, "upper", local_uv, data.uv_faces, tolerance)

    # All center positions are half-spacing inset from the two waist references.
    center_inner_x = center_x
    (
        center_inner,
        rejected_center_boundary_tracks,
        rejected_center_discontinuous_tracks,
    ) = _available_center_tracks(
        center_inner_x, local_uv, data.uv_faces, tolerance
    )
    paired_lower = max(len(lower_left), len(lower_right))
    paired_upper = max(len(upper_left), len(upper_right))
    if not len(center_inner):
        raise ValueError("The waist is too narrow to form an interior central scan.")

    segments: list[TrajectorySegment] = []
    last_surface: TrajectorySegment | None = None
    lower_accepted_x: list[float] = []
    upper_accepted_x: list[float] = []
    rejected_lower_spacing_tracks = 0
    rejected_upper_spacing_tracks = 0

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

    def append_jump(source_uv: np.ndarray, from_a: bool) -> np.ndarray:
        destination_uv = _seam_counterpart(source_uv, seam_segments, from_a=from_a)
        jump_uv = np.vstack([source_uv, destination_uv])
        segments.append(TrajectorySegment(
            kind="SEAM_JUMP",
            uv=jump_uv,
            xyz=mapper.map(jump_uv),
            spray_on=True,
            note="Two UV coordinates represent one physical seam point",
        ))
        return destination_uv

    for index in range(paired_lower):
        left_available = index < len(lower_left)
        right_available = index < len(lower_right)

        # Alternate starting side for better coverage balance
        prefer_left_first = index % 2 == 0

        if prefer_left_first and left_available:
            first_track = lower_left[index]
            first_is_a = True
            labels = ("lower-left upward", "lower-right downward (seam-driven)")
        elif not prefer_left_first and right_available:
            first_track = lower_right[index]
            first_is_a = False
            labels = ("lower-right upward", "lower-left downward (seam-driven)")
        elif left_available:
            first_track = lower_left[index]
            first_is_a = True
            labels = ("lower-left upward (unpaired)", "lower-right downward (seam-driven)")
        elif right_available:
            first_track = lower_right[index]
            first_is_a = False
            labels = ("lower-right upward (unpaired)", "lower-left downward (seam-driven)")
        else:
            continue  # Should never happen given paired_lower = max(...)

        outer_snap_tolerance = boundary_snap_tolerance if index == 0 else None
        first = _make_surface_scan(
            *first_track, True, sample_step, frame, mapper,
            snap_tolerance=outer_snap_tolerance, note=labels[0]
        )
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=first_is_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(
            first.uv, outer_snap_tolerance, bridge_outside=True
        )

        second = _make_scan_from_seam_destination(
            destination, "lower", False, sample_step,
            frame, local_uv, data.uv_faces, tolerance, mapper,
            outer_snap_tolerance, note=labels[1],
        )
        if second is not None:
            second_x = float(_to_local(second.uv[:1], frame)[0, 0])
            append_surface(first, connect_from_previous=(last_surface is not None))
            lower_accepted_x.append(float(first_track[0]))
            append_jump(first.uv[-1], from_a=first_is_a)
            append_surface(second, connect_from_previous=False)
            lower_accepted_x.append(second_x)
        else:
            # Preserve the available half instead of deleting the whole offset.
            append_surface(first, connect_from_previous=(last_surface is not None))
            lower_accepted_x.append(float(first_track[0]))

    # Choose the centered waist-track order from the lower endpoint nearest to
    # the preceding scan.
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
        ascending_gap = float(np.linalg.norm(
            last_surface.xyz[-1] - ascending_first.xyz[0]
        ))
        descending_gap = float(np.linalg.norm(
            last_surface.xyz[-1] - descending_first.xyz[0]
        ))
        if descending_gap < ascending_gap:
            inner_tracks = descending_tracks

    for index, track in enumerate(inner_tracks):
        scan = _make_surface_scan(
            *track, upward=(index % 2 == 0), sample_step=sample_step,
            frame=frame, mapper=mapper, note="central bow scan",
        )
        append_surface(
            scan,
            connect_from_previous=(last_surface is not None),
            region_transition=(index == 0 and last_surface is not None),
        )

    # The odd central-track count exits at the upper opening. Start the upper
    # region on the nearest seam side.
    upper_start_is_a = bool(inner_tracks[-1][0] <= 0.5 * frame.waist_width)

    # At the upper waist, work outwards. A downward pass reaches one seam copy,
    # then resumes upward from the equivalent point on the other copy.
    next_upper_start_is_a = upper_start_is_a
    for index in range(paired_upper):
        left_available = index < len(upper_left)
        right_available = index < len(upper_right)

        # Continue from the same UV copy on which the previous complete pair ended.
        if not next_upper_start_is_a and right_available:
            first_track = upper_right[index]
            first_is_a = False
            labels = ("upper-right downward", "upper-left upward (seam-driven)")
        elif next_upper_start_is_a and left_available:
            first_track = upper_left[index]
            first_is_a = True
            labels = ("upper-left downward", "upper-right upward (seam-driven)")
        else:
            # The shared count normally keeps both sides available.  Do not
            # silently reverse sides here: that would break the one-stroke
            # topology at the opening.
            continue

        outer_snap_tolerance = boundary_snap_tolerance if index == paired_upper - 1 else None
        first = _make_surface_scan(
            *first_track, False, sample_step, frame, mapper,
            snap_tolerance=outer_snap_tolerance, note=labels[0]
        )
        # The triangle-union interval endpoint can sit a few UV units inside
        # the saved cut edge.  Force it onto that edge and obtain the opposite
        # UV coordinate using the very same physical interpolation parameter.
        source, destination = _seam_projection_and_counterpart(
            first.uv[-1], seam_segments, from_a=first_is_a
        )
        first.uv[-1] = source
        first.xyz = mapper.map(
            first.uv, outer_snap_tolerance, bridge_outside=True
        )
        second = _make_scan_from_seam_destination(
            destination, "upper", True, sample_step,
            frame, local_uv, data.uv_faces, tolerance, mapper,
            outer_snap_tolerance, note=labels[1],
        )
        if second is not None:
            second_x = float(_to_local(second.uv[:1], frame)[0, 0])
        else:
            # Keep an available offset even when no counterpart interval can
            # be constructed at this seam destination.
            append_surface(
                first,
                connect_from_previous=(last_surface is not None),
                region_transition=(not upper_accepted_x and last_surface is not None),
            )
            upper_accepted_x.append(float(first_track[0]))
            continue

        # Commit both halves as one complete seam pair.
        append_surface(
            first,
            connect_from_previous=(last_surface is not None),
            region_transition=(not upper_accepted_x and last_surface is not None),
        )
        append_jump(first.uv[-1], from_a=first_is_a)
        append_surface(second, connect_from_previous=False)
        upper_accepted_x.extend((float(first_track[0]), second_x))
        next_upper_start_is_a = not first_is_a

    segments = _smooth_seam_jumps_xyz(
        segments, nominal_radius=0.5 * target_spacing, sample_step=sample_step,
    )
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
            "lower_left": lower_left_spacing,
            "lower_right": lower_right_spacing,
            "center": center_spacing,
            "upper_left": upper_left_spacing,
            "upper_right": upper_right_spacing,
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


def _show_result(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    cut_edges: np.ndarray | None = None,
    trajectory: Sequence[TrajectorySegment] | None = None,
    planning_frame: PlanningFrame | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Render the final trajectory as batched VTK/OpenGL geometry."""
    from vtktrajdisplay import show_result

    show_result(
        mesh, uv, uv_faces, cut_edges,
        trajectory=trajectory, planning_frame=planning_frame, metadata=metadata,
    )


def _request_spacing(
    mesh: trimesh.Trimesh,
    data: ObjUVData,
    seam: PrecutSeam,
    cut_edges: np.ndarray,
    initial_spacing: float,
) -> tuple[float | None, PlanningFrame]:
    """Show the UV result without trajectories and request nominal spacing."""
    from vtktrajdisplay import request_spacing

    frame = _planning_frame(_resolve_main_seam_uv_segments(data, seam))
    spacing = request_spacing(
        mesh, data.uv, data.uv_faces, initial_spacing,
        cut_edges=cut_edges, planning_frame=frame,
    )
    return spacing, frame


def _play_iteration_animation(mesh: trimesh.Trimesh, result_obj: Path) -> None:
    """Render OptCuts iterations with a VTK/OpenGL timer pipeline."""
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
    parser.add_argument("--no-animation", action="store_true", help="Skip the OptCuts iteration animation.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the final VTK/OpenGL window.")
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
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
    else:
        selected_spacing, _ = _request_spacing(
            mesh, data, precut_seam, cut_edges, args.spacing
        )
        if selected_spacing is None:
            print("Trajectory generation cancelled.")
            return
    sample_step = None if args.sample_step <= 0.0 else args.sample_step
    trajectory, planning_frame, metadata = generate_seam_aware_trajectory(
        data, precut_seam, target_spacing=selected_spacing, sample_step=sample_step,
    )
    selected_path = Path(selected)
    csv_path = Path(args.trajectory_out) if args.trajectory_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.csv"
    )
    metadata_path = Path(args.metadata_out) if args.metadata_out else selected_path.with_name(
        selected_path.stem + "_optcuts_trajectory.json"
    )
    _export_trajectory(csv_path, metadata_path, trajectory, metadata)
    print(f"Pre-cut shortest-boundary seam edges: {len(precut_seam.edges)}")
    print(f"Additional detected OptCuts seam edges: {len(optimized_cut_edges)}")
    print(f"Trajectory events: {len(trajectory)}")
    if not args.no_show:
        _show_result(
            mesh, data.uv, data.uv_faces, cut_edges,
            trajectory=trajectory, planning_frame=planning_frame, metadata=metadata,
        )
    if not args.no_animation:
        _play_iteration_animation(optcuts_mesh, result_obj)


if __name__ == "__main__":
    main()
