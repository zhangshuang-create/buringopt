from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.interpolate as interpolate
from scipy.spatial import cKDTree
import trimesh
import vtk

EPS = 1.0e-9
VALID_EVENTS = {"SURFACE_SCAN", "OVERTRAVEL", "SEAM_JUMP", "SEAM_BLEND_3D"}


# =========================================================================
# 一、 极速向量化 3D 相交与局部切分算法 (KD-Tree 邻域剪枝)
# =========================================================================


def _segment_closest_dist_3d(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> Tuple[float, float]:
    """计算两线段空间最近距离及 p1-p2 上的比例参数 u。"""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-9 or e <= 1e-9:
        return float(np.linalg.norm(r)), 0.0

    b = float(np.dot(d1, d2))
    c = float(np.dot(d1, r))
    denom = a * e - b * b

    if abs(denom) > 1e-9:
        u = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
    else:
        u = 0.0

    v = (b * u + f) / e
    if v < 0.0:
        v = 0.0
        u = float(np.clip(-c / a, 0.0, 1.0))
    elif v > 1.0:
        v = 1.0
        u = float(np.clip((b - c) / a, 0.0, 1.0))

    p_close = p1 + u * d1
    q_close = q1 + v * d2
    return float(np.linalg.norm(p_close - q_close)), u


def fast_partition_active_track(
    active_xyz: np.ndarray,
    other_kdtree: Optional[cKDTree],
    other_segments: np.ndarray,  # shape: (M, 2, 3) 其它所有线段端点
    tol: float = 0.8,
) -> List[Tuple[np.ndarray, bool]]:
    """
    基于全局空间哈希的极速求交：
    1. 用 KDTree 过滤出候选干涉线段；
    2. 局部高精求交并切分；
    3. 耗时降至 1~2 毫秒。
    """
    if len(active_xyz) < 2 or other_kdtree is None or len(other_segments) == 0:
        return [(active_xyz, False)]

    # 1. 批量球查询快速锁定可能相交的线段
    candidate_indices = other_kdtree.query_ball_point(active_xyz, r=tol * 2.0)

    # 2. 沿活动线插值交点
    refined_pts = [active_xyz[0]]
    for i in range(len(active_xyz) - 1):
        p1, p2 = active_xyz[i], active_xyz[i + 1]

        # 获取 p1 和 p2 周围候选线段并去重
        local_cand = set(candidate_indices[i]) | set(candidate_indices[i + 1])
        intersections = []

        for seg_idx in local_cand:
            q1, q2 = other_segments[seg_idx]
            dist, u = _segment_closest_dist_3d(p1, p2, q1, q2)
            if dist <= tol and (1e-4 < u < 1.0 - 1e-4):
                intersections.append((u, p1 + u * (p2 - p1)))

        intersections.sort(key=lambda x: x[0])
        for _, pt in intersections:
            refined_pts.append(pt)
        refined_pts.append(p2)

    refined_pts_arr = np.asarray(refined_pts, dtype=float)

    # 3. 中点干涉状态判定
    mid_pts = 0.5 * (refined_pts_arr[:-1] + refined_pts_arr[1:])
    dists, _ = other_kdtree.query(mid_pts)
    seg_hit = dists <= tol * 1.2

    # 4. 聚合成子段
    sub_sections: List[Tuple[np.ndarray, bool]] = []
    curr_state = bool(seg_hit[0])
    curr_pts = [refined_pts_arr[0]]

    for i in range(len(seg_hit)):
        curr_pts.append(refined_pts_arr[i + 1])
        if i == len(seg_hit) - 1 or seg_hit[i + 1] != curr_state:
            sub_sections.append((np.asarray(curr_pts, dtype=float), curr_state))
            if i < len(seg_hit) - 1:
                curr_state = bool(seg_hit[i + 1])
                curr_pts = [refined_pts_arr[i + 1]]

    return sub_sections


# =========================================================================
# 二、 轨迹数据结构与文件解析
# =========================================================================


@dataclass
class TrackSegmentData:
    segment_id: int
    event: str
    spray_on: bool
    region: str
    note: str
    dense_xyz: np.ndarray
    control_points: np.ndarray
    mode: str = "SPLINE"


def _as_points(value: object, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{label} 必须包含有效的三维坐标。")
    return points


def load_trajectory_file(
    csv_path: Path, metadata_path: Optional[Path] = None
) -> List[TrackSegmentData]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    segments_raw: Dict[int, list] = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])

        req_core = {"x", "y", "z"}
        if not req_core.issubset(fields):
            raise ValueError(f"CSV 格式不合法，缺少必须的坐标列 (x, y, z): {fields}")

        for row_idx, row in enumerate(reader):
            seg_id = int(row.get("segment_id", 0))
            pt_id = int(row.get("point_id", row_idx))
            event = str(row.get("event", "SURFACE_SCAN")).strip().upper()
            spray_on_raw = str(row.get("spray_on", "1")).strip().lower()
            spray_on = spray_on_raw in {"1", "true", "yes"}
            region = str(row.get("region", "CENTER")).strip().upper()
            note = str(row.get("note", ""))
            pt = [float(row["x"]), float(row["y"]), float(row["z"])]

            segments_raw.setdefault(seg_id, []).append(
                (pt_id, event, spray_on, region, pt, note)
            )

    control_by_segment: Dict[int, np.ndarray] = {}
    json_path = metadata_path if metadata_path else csv_path.with_suffix(".json")
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for item in meta.get("control_points", []):
                control_by_segment[int(item["segment_id"])] = _as_points(
                    item["points"], "control_points"
                )
        except Exception:
            pass

    segments: List[TrackSegmentData] = []
    for seg_id in sorted(segments_raw):
        rows = sorted(segments_raw[seg_id], key=lambda x: x[0])
        _, event, spray_on, region, _, note = rows[0]
        pts = np.asarray([r[4] for r in rows], dtype=float)

        if seg_id in control_by_segment:
            ctrls = control_by_segment[seg_id].copy()
        else:
            ctrls = pts.copy()

        segments.append(
            TrackSegmentData(
                segment_id=seg_id,
                event=event,
                spray_on=spray_on,
                region=region,
                note=note,
                dense_xyz=pts,
                control_points=ctrls,
            )
        )

    return segments


# =========================================================================
# 三、 多段流形拟合引擎 (带空间静态加速缓存)
# =========================================================================


class MultiSegmentSplineEngine:
    def __init__(self, mesh: trimesh.Trimesh, sample_step: float = 1.0):
        self.mesh = mesh
        self.sample_step = max(float(sample_step), 0.2)
        self.segments: List[TrackSegmentData] = []
        self.active_segment_idx: Optional[int] = None
        self.show_all_segments: bool = True

        # 空间检索加速缓存
        self.other_kdtree: Optional[cKDTree] = None
        self.other_segment_lines: np.ndarray = np.empty((0, 2, 3), dtype=float)

    def initialize_segments(
        self, seg_list: List[TrackSegmentData], default_ctrl_count: int = 16
    ):
        self.segments = seg_list
        for seg in self.segments:
            if seg.event == "SURFACE_SCAN" and len(seg.dense_xyz) >= 2:
                if len(seg.control_points) < 4 or len(seg.control_points) == len(
                    seg.dense_xyz
                ):
                    seg.control_points = self._fit_initial_control_points(
                        seg.dense_xyz, default_ctrl_count
                    )
                seg.dense_xyz = self._eval_surface_spline(
                    seg.control_points, mode=seg.mode
                )
            else:
                seg.control_points = (
                    np.vstack([seg.dense_xyz[0], seg.dense_xyz[-1]])
                    if len(seg.dense_xyz) >= 2
                    else seg.dense_xyz.copy()
                )

        self._rebuild_all_transitions()
        self.rebuild_spatial_cache()

    def rebuild_spatial_cache(self):
        """为所有非活动段预构建静态检索空间树（仅在切换选择或载入时计算一次）。"""
        if self.active_segment_idx is None:
            self.other_kdtree = None
            self.other_segment_lines = np.empty((0, 2, 3), dtype=float)
            return

        lines_list = []
        pts_list = []
        for i, s in enumerate(self.segments):
            if (
                i == self.active_segment_idx
                or s.event != "SURFACE_SCAN"
                or len(s.dense_xyz) < 2
            ):
                continue
            pts = s.dense_xyz
            pts_list.append(pts)
            # 构建线段列表 (N-1, 2, 3)
            lines_list.append(np.stack([pts[:-1], pts[1:]], axis=1))

        if lines_list:
            self.other_segment_lines = np.vstack(lines_list)
            all_pts = np.vstack(pts_list)
            # 使用线段中点加速查询
            seg_midpoints = np.mean(self.other_segment_lines, axis=1)
            self.other_kdtree = cKDTree(seg_midpoints)
        else:
            self.other_kdtree = None
            self.other_segment_lines = np.empty((0, 2, 3), dtype=float)

    def _clean_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.r_[True, dists > 1e-4]
        cleaned = pts[keep]
        return cleaned if len(cleaned) >= 2 else pts

    def _fit_initial_control_points(
        self, polyline: np.ndarray, num_ctrl: int
    ) -> np.ndarray:
        pts = self._clean_points(polyline)
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dists = np.maximum(dists, 1e-4)
        cum = np.r_[0.0, np.cumsum(dists)]
        total = max(cum[-1], 1e-5)

        n_ctrl = max(4, min(num_ctrl, len(pts)))
        u_samples = np.linspace(0.0, total, n_ctrl)
        ctrl_init = np.empty((n_ctrl, 3), dtype=float)
        for dim in range(3):
            ctrl_init[:, dim] = np.interp(u_samples, cum, pts[:, dim])

        proj_ctrl, _, _ = trimesh.proximity.closest_point(self.mesh, ctrl_init)
        proj_ctrl[0] = pts[0]
        proj_ctrl[-1] = pts[-1]
        return proj_ctrl

    def _eval_surface_spline(
        self, ctrl_pts: np.ndarray, mode: str = "SPLINE"
    ) -> np.ndarray:
        pts = self._clean_points(ctrl_pts)
        n = len(pts)
        if n < 2:
            return pts.copy()

        if mode == "LINE" or n < 3:
            return self._eval_linear_surface(pts)

        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dists = np.maximum(dists, 1e-4)
        cum = np.r_[0.0, np.cumsum(dists)]
        total = max(cum[-1], 1e-5)
        u_knots = cum / total

        for i in range(1, len(u_knots)):
            if u_knots[i] <= u_knots[i - 1]:
                u_knots[i] = u_knots[i - 1] + 1e-4
        u_knots /= u_knots[-1]

        k_deg = min(3, n - 1)
        try:
            tck, _ = interpolate.splprep(
                [pts[:, 0], pts[:, 1], pts[:, 2]], u=u_knots, k=k_deg, s=0.0
            )
            n_eval = max(24, int(np.ceil(total / self.sample_step)))
            u_eval = np.linspace(0.0, 1.0, n_eval)
            eval_pts = np.column_stack(interpolate.splev(u_eval, tck))

            projected, _, _ = trimesh.proximity.closest_point(self.mesh, eval_pts)
            projected[0] = pts[0]
            projected[-1] = pts[-1]
            return projected
        except Exception:
            return self._eval_linear_surface(pts)

    def _eval_linear_surface(self, pts: np.ndarray) -> np.ndarray:
        curve_segments = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dist = float(np.linalg.norm(p1 - p0))
            num_samples = max(4, int(np.ceil(dist / self.sample_step)))
            t_vals = np.linspace(0.0, 1.0, num_samples)
            seg_3d = (1.0 - t_vals[:, None]) * p0 + t_vals[:, None] * p1
            seg_proj, _, _ = trimesh.proximity.closest_point(self.mesh, seg_3d)
            seg_proj[0], seg_proj[-1] = p0, p1
            curve_segments.append(seg_proj if not curve_segments else seg_proj[1:])
        return np.vstack(curve_segments) if curve_segments else pts.copy()

    def update_control_point(self, seg_idx: int, ctrl_idx: int, new_xyz: np.ndarray):
        if not (0 <= seg_idx < len(self.segments)):
            return

        seg = self.segments[seg_idx]
        if not (0 <= ctrl_idx < len(seg.control_points)):
            return

        proj_pt, _, _ = trimesh.proximity.closest_point(self.mesh, new_xyz[None, :])
        seg.control_points[ctrl_idx] = proj_pt[0]
        seg.dense_xyz = self._eval_surface_spline(seg.control_points, mode=seg.mode)

        if ctrl_idx == 0 or ctrl_idx == len(seg.control_points) - 1:
            self._rebuild_neighbor_transitions(seg_idx)

    def _c1_connector(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        t_start: np.ndarray,
        t_end: np.ndarray,
    ) -> np.ndarray:
        gap = float(np.linalg.norm(p_end - p_start))
        if gap <= 1.0e-7:
            return np.vstack([p_start, p_end])

        t0 = t_start / max(float(np.linalg.norm(t_start)), 1.0e-9)
        t1 = t_end / max(float(np.linalg.norm(t_end)), 1.0e-9)
        handle = 0.45 * gap
        n_samples = max(4, int(np.ceil(gap / self.sample_step)) + 1)
        u = np.linspace(0.0, 1.0, n_samples)[:, None]

        curve = (
            (2 * u**3 - 3 * u**2 + 1) * p_start
            + (u**3 - 2 * u**2 + u) * handle * t0
            + (-2 * u**3 + 3 * u**2) * p_end
            + (u**3 - u**2) * handle * t1
        )
        return curve

    def _rebuild_neighbor_transitions(self, seg_idx: int):
        if seg_idx > 0 and self.segments[seg_idx - 1].event in {
            "OVERTRAVEL",
            "SEAM_JUMP",
        }:
            prev_seg = self.segments[seg_idx - 1]
            curr_seg = self.segments[seg_idx]
            p0 = prev_seg.dense_xyz[0]
            p1 = curr_seg.dense_xyz[0]
            t0 = (
                (prev_seg.dense_xyz[1] - prev_seg.dense_xyz[0])
                if len(prev_seg.dense_xyz) >= 2
                else (p1 - p0)
            )
            t1 = (
                (curr_seg.dense_xyz[1] - curr_seg.dense_xyz[0])
                if len(curr_seg.dense_xyz) >= 2
                else (p1 - p0)
            )
            prev_seg.dense_xyz = self._c1_connector(p0, p1, t0, t1)
            prev_seg.control_points = np.vstack(
                [prev_seg.dense_xyz[0], prev_seg.dense_xyz[-1]]
            )

        if seg_idx < len(self.segments) - 1 and self.segments[seg_idx + 1].event in {
            "OVERTRAVEL",
            "SEAM_JUMP",
        }:
            next_seg = self.segments[seg_idx + 1]
            curr_seg = self.segments[seg_idx]
            p0 = curr_seg.dense_xyz[-1]
            p1 = next_seg.dense_xyz[-1]
            t0 = (
                (curr_seg.dense_xyz[-1] - curr_seg.dense_xyz[-2])
                if len(curr_seg.dense_xyz) >= 2
                else (p1 - p0)
            )
            t1 = (
                (next_seg.dense_xyz[-1] - next_seg.dense_xyz[-2])
                if len(next_seg.dense_xyz) >= 2
                else (p1 - p0)
            )
            next_seg.dense_xyz = self._c1_connector(p0, p1, t0, t1)
            next_seg.control_points = np.vstack(
                [next_seg.dense_xyz[0], next_seg.dense_xyz[-1]]
            )

    def _rebuild_all_transitions(self):
        for i in range(len(self.segments)):
            if self.segments[i].event == "SURFACE_SCAN":
                self._rebuild_neighbor_transitions(i)


# =========================================================================
# 四、 VTK 交互拖拽器
# =========================================================================


class SurfaceTrackInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, app: SurfaceCurveApp):
        super().__init__()
        self.app = app
        self.is_dragging: bool = False
        self.drag_ctrl_idx: Optional[int] = None

        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.005)
        self.cell_picker.AddPickList(self.app.mesh_actor)
        self.cell_picker.PickFromListOn()

        self.prop_picker = vtk.vtkPropPicker()

        self.AddObserver("LeftButtonPressEvent", self.on_left_down)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_up)

    def on_left_down(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        renderer = self.app.renderer

        self.prop_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        actor = self.prop_picker.GetActor()

        if actor is not None and actor in self.app.handle_actor_map:
            ctrl_idx = self.app.handle_actor_map[actor]
            self.drag_ctrl_idx = ctrl_idx
            self.is_dragging = True
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            self.app.render()
            return

        self.OnLeftButtonDown()

    def on_mouse_move(self, obj, event):
        if (
            self.is_dragging
            and self.drag_ctrl_idx is not None
            and self.app.engine.active_segment_idx is not None
        ):
            mouse_pos = self.GetInteractor().GetEventPosition()
            renderer = self.app.renderer

            if self.cell_picker.Pick(mouse_pos[0], mouse_pos[1], 0, renderer):
                surf_pos = np.array(self.cell_picker.GetPickPosition())
                self.app.engine.update_control_point(
                    self.app.engine.active_segment_idx, self.drag_ctrl_idx, surf_pos
                )
                self.app.update_active_segment_visuals()
            return

        self.OnMouseMove()

    def on_left_up(self, obj, event):
        if self.is_dragging:
            self.is_dragging = False
            self.drag_ctrl_idx = None
            self.app.refresh_handles()
            self.app.render()
        else:
            self.OnLeftButtonUp()


# =========================================================================
# 五、 VTK 主渲染应用
# =========================================================================


class SurfaceCurveApp:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.engine = MultiSegmentSplineEngine(mesh)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.14, 0.18)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetWindowName(
            "3D Mesh Trajectory Designer - [Fast Intersection Mode]"
        )
        self.render_window.SetSize(1450, 900)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.mesh_actor = self._create_mesh_actor(mesh)
        self.renderer.AddActor(self.mesh_actor)

        self.track_actors: Dict[int, vtk.vtkActor] = {}
        self.active_segment_sub_actors: List[vtk.vtkActor] = []

        self.handle_actors: List[vtk.vtkActor] = []
        self.handle_actor_map: Dict[vtk.vtkActor, int] = {}

        self.info_text_actor = vtk.vtkTextActor()
        self.info_text_actor.SetPosition(20, 20)
        self.info_text_actor.GetTextProperty().SetFontSize(15)
        self.info_text_actor.GetTextProperty().SetColor(0.85, 0.92, 1.0)
        self.renderer.AddViewProp(self.info_text_actor)

        self.style = SurfaceTrackInteractorStyle(self)
        self.interactor.SetInteractorStyle(self.style)

        self.panel: Optional[TkControlPanel] = None

    def _create_mesh_actor(self, mesh: trimesh.Trimesh) -> vtk.vtkActor:
        pts = vtk.vtkPoints()
        for v in mesh.vertices:
            pts.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))

        cells = vtk.vtkCellArray()
        for f in mesh.faces:
            cells.InsertNextCell(3)
            cells.InsertCellPoint(int(f[0]))
            cells.InsertCellPoint(int(f[1]))
            cells.InsertCellPoint(int(f[2]))

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.68, 0.74, 0.82)
        actor.GetProperty().SetOpacity(0.92)
        actor.GetProperty().SetSpecular(0.2)
        return actor

    def update_all_visuals(self):
        for actor in self.track_actors.values():
            self.renderer.RemoveActor(actor)
        self.track_actors.clear()

        for seg_idx in range(len(self.engine.segments)):
            self._create_or_update_track_actor(seg_idx)

        self._render_active_segment_partitioned()
        self.refresh_handles()
        self._update_text()
        self.render()

    def update_active_segment_visuals(self):
        curr_idx = self.engine.active_segment_idx
        if curr_idx is not None:
            if curr_idx > 0:
                self._create_or_update_track_actor(curr_idx - 1)
            if curr_idx < len(self.engine.segments) - 1:
                self._create_or_update_track_actor(curr_idx + 1)

            self._render_active_segment_partitioned()

            seg = self.engine.segments[curr_idx]
            for actor, c_i in list(self.handle_actor_map.items()):
                if c_i < len(seg.control_points):
                    pt = seg.control_points[c_i]
                    actor.SetPosition(float(pt[0]), float(pt[1]), float(pt[2]))

        self._update_text()
        self.render()

    def _create_polyline_actor(
        self, pts: np.ndarray, color: tuple, width: float
    ) -> vtk.vtkActor:
        vtk_pts = vtk.vtkPoints()
        for p in pts:
            vtk_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(pts))
        for i in range(len(pts)):
            lines.InsertCellPoint(i)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        try:
            mapper.SetResolveCoincidentTopologyLineOffsetParameters(-2.0, -2.0)
        except AttributeError:
            pass

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRenderLinesAsTubes(True)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        return actor

    def _render_active_segment_partitioned(self):
        """【毫秒级极速着色】：仅计算活动段自身交点，相交段变黄，安全段变绿。"""
        for a in self.active_segment_sub_actors:
            self.renderer.RemoveActor(a)
        self.active_segment_sub_actors.clear()

        curr_idx = self.engine.active_segment_idx
        if curr_idx is None or not (0 <= curr_idx < len(self.engine.segments)):
            return

        active_seg = self.engine.segments[curr_idx]
        if len(active_seg.dense_xyz) < 2:
            return

        # 极速空间检索分段
        sub_sections = fast_partition_active_track(
            active_seg.dense_xyz,
            self.engine.other_kdtree,
            self.engine.other_segment_lines,
            tol=0.8,
        )

        for pts, is_intersected in sub_sections:
            if len(pts) < 2:
                continue
            if is_intersected:
                # 仅交点区间高亮亮黄色
                sub_actor = self._create_polyline_actor(
                    pts, color=(1.0, 0.95, 0.0), width=6.0
                )
            else:
                # 未相交部分保持亮绿色
                sub_actor = self._create_polyline_actor(
                    pts, color=(0.0, 1.0, 0.4), width=4.5
                )

            self.renderer.AddActor(sub_actor)
            self.active_segment_sub_actors.append(sub_actor)

    def _create_or_update_track_actor(self, seg_idx: int):
        if (
            self.engine.active_segment_idx is not None
            and seg_idx == self.engine.active_segment_idx
        ):
            if seg_idx in self.track_actors:
                self.track_actors[seg_idx].SetVisibility(False)
            return

        seg = self.engine.segments[seg_idx]
        pts = seg.dense_xyz
        if len(pts) < 2:
            return

        if seg.event == "SURFACE_SCAN":
            color = (0.88, 0.12, 0.12)  # 邻居表面线：红色
            width = 3.0
        else:
            color = (0.1, 0.8, 0.2)  # 过渡段：绿色
            width = 2.0

        if seg_idx in self.track_actors:
            self.renderer.RemoveActor(self.track_actors[seg_idx])

        actor = self._create_polyline_actor(pts, color=color, width=width)

        if not self.engine.show_all_segments:
            actor.SetVisibility(False)
        else:
            actor.SetVisibility(True)

        self.renderer.AddActor(actor)
        self.track_actors[seg_idx] = actor

    def refresh_handles(self):
        for actor in self.handle_actors:
            self.renderer.RemoveActor(actor)
        self.handle_actors.clear()
        self.handle_actor_map.clear()

        if self.engine.active_segment_idx is None:
            return
        if not (0 <= self.engine.active_segment_idx < len(self.engine.segments)):
            return

        seg = self.engine.segments[self.engine.active_segment_idx]
        if seg.event != "SURFACE_SCAN":
            return

        scale = float(np.max(self.mesh.extents))
        r_sphere = max(scale * 0.012, 0.8)

        for ctrl_idx, pt in enumerate(seg.control_points):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(r_sphere)
            sphere.SetThetaResolution(14)
            sphere.SetPhiResolution(14)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPosition(float(pt[0]), float(pt[1]), float(pt[2]))

            if ctrl_idx == 0 or ctrl_idx == len(seg.control_points) - 1:
                actor.GetProperty().SetColor(0.2, 0.95, 0.3)
            else:
                actor.GetProperty().SetColor(1.0, 0.5, 0.0)

            self.renderer.AddActor(actor)
            self.handle_actors.append(actor)
            self.handle_actor_map[actor] = ctrl_idx

    def select_segment(self, seg_idx: Optional[int]):
        self.engine.active_segment_idx = seg_idx
        # 仅在切换选择时重建一次 KD-Tree 缓存
        self.engine.rebuild_spatial_cache()
        self.update_all_visuals()

    def set_display_mode(self, show_all: bool):
        self.engine.show_all_segments = show_all
        for seg_idx, actor in self.track_actors.items():
            if not show_all:
                actor.SetVisibility(False)
            else:
                actor.SetVisibility(True)
        self.render()

    def _update_text(self):
        if not self.engine.segments:
            self.info_text_actor.SetInput("请在控制台加载轨迹 CSV 文件")
            return

        if self.engine.active_segment_idx is None:
            self.info_text_actor.SetInput(
                f"未选中任何轨迹（不执行相交判定） | 总段数: {len(self.engine.segments)}"
            )
            return

        seg = self.engine.segments[self.engine.active_segment_idx]
        mode_name = (
            "五次/三次 B 样条 (Spline)" if seg.mode == "SPLINE" else "曲面直线段 (Line)"
        )
        disp_name = (
            "【显示全部轨迹】"
            if self.engine.show_all_segments
            else "【仅显示选中段 (Solo)】"
        )

        info = (
            f"当前活动段: Segment #{seg.segment_id} [{seg.event} - {seg.region}] | 总段数: {len(self.engine.segments)}\n"
            f"显示模式: {disp_name} | 曲线模式: {mode_name}\n"
            f"控制点数: {len(seg.control_points)} | 采样点数: {len(seg.dense_xyz)}\n"
            f"提示: 空间加速索引已开启；拖拽时仅活动轨迹产生相交的区间实时变黄，邻居保持原色。"
        )
        self.info_text_actor.SetInput(info)

    def _reset_camera(self):
        center = self.mesh.vertices.mean(axis=0)
        dist = max(float(np.linalg.norm(np.ptp(self.mesh.vertices, axis=0))), 1.0)
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(center[0], center[1] - 2.2 * dist, center[2] + 0.8 * dist)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()

    def render(self):
        self.render_window.Render()


# =========================================================================
# 六、 控制面板与主入口
# =========================================================================


class TkControlPanel:
    def __init__(self, app: SurfaceCurveApp, tk_root: tk.Tk):
        self.app = app
        self.root = tk_root
        self.root.title("轨迹分段编辑与显示控制台")
        self.root.geometry("420x540+40+40")
        self.root.wm_attributes("-topmost", True)
        self.root.deiconify()

        # 1. 文件区
        f_frame = tk.LabelFrame(self.root, text="文件载入与导出", padx=8, pady=6)
        f_frame.pack(fill="x", padx=10, pady=5)

        btn_row = tk.Frame(f_frame)
        btn_row.pack(fill="x")
        tk.Button(
            btn_row,
            text="打开轨迹 CSV",
            command=self.open_csv_dialog,
            bg="#1948CD",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(
            btn_row,
            text="保存导出 (CSV/PAQ)",
            command=self.export_dialog,
            bg="#008040",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side="right", expand=True, fill="x", padx=2)

        # 2. 轨迹段选择器
        seg_frame = tk.LabelFrame(
            self.root, text="轨迹段切换 (Segment Selector)", padx=8, pady=6
        )
        seg_frame.pack(fill="x", padx=10, pady=5)

        combo_row = tk.Frame(seg_frame)
        combo_row.pack(fill="x", pady=2)
        tk.Label(combo_row, text="目标轨迹段:").pack(side="left")
        self.combo_var = tk.StringVar()
        self.seg_combobox = ttk.Combobox(
            combo_row, textvariable=self.combo_var, state="readonly"
        )
        self.seg_combobox.pack(side="right", expand=True, fill="x", padx=5)
        self.seg_combobox.bind("<<ComboboxSelected>>", self.on_seg_combo_changed)

        nav_row = tk.Frame(seg_frame)
        nav_row.pack(fill="x", pady=4)
        tk.Button(nav_row, text="◀ 上一段 (Prev)", command=self.prev_segment).pack(
            side="left", expand=True, fill="x", padx=2
        )
        tk.Button(nav_row, text="下一段 (Next) ▶", command=self.next_segment).pack(
            side="left", expand=True, fill="x", padx=2
        )
        tk.Button(nav_row, text="取消选中", command=self.deselect_segment).pack(
            side="right", expand=True, fill="x", padx=2
        )

        # 3. 显示与过滤模式
        disp_frame = tk.LabelFrame(
            self.root, text="显示与过滤模式 (Display & Filter)", padx=8, pady=6
        )
        disp_frame.pack(fill="x", padx=10, pady=5)

        self.show_all_var = tk.BooleanVar(value=True)
        self.cb_show_all = tk.Checkbutton(
            disp_frame,
            text="全显示模式 (勾选: 显示全部; 取消: 仅显示当前段)",
            variable=self.show_all_var,
            command=self.on_show_all_toggled,
            font=("Arial", 9, "bold"),
        )
        self.cb_show_all.pack(anchor="w", pady=2)

        self.show_mesh_var = tk.BooleanVar(value=True)
        self.cb_mesh = tk.Checkbutton(
            disp_frame,
            text="显示 3D 网格模型 (Mesh)",
            variable=self.show_mesh_var,
            command=self.on_mesh_toggled,
        )
        self.cb_mesh.pack(anchor="w", pady=2)

        # 4. 当前段样条参数调节
        param_frame = tk.LabelFrame(self.root, text="当前段样条参数", padx=8, pady=6)
        param_frame.pack(fill="x", padx=10, pady=5)

        p_row1 = tk.Frame(param_frame)
        p_row1.pack(fill="x", pady=2)
        tk.Label(p_row1, text="绘制模式:").pack(side="left")
        self.mode_var = tk.StringVar(value="SPLINE")
        tk.Radiobutton(
            p_row1,
            text="五次/三次 B 样条",
            variable=self.mode_var,
            value="SPLINE",
            command=self.on_mode_changed,
        ).pack(side="left", padx=5)
        tk.Radiobutton(
            p_row1,
            text="直线段",
            variable=self.mode_var,
            value="LINE",
            command=self.on_mode_changed,
        ).pack(side="left", padx=5)

        p_row2 = tk.Frame(param_frame)
        p_row2.pack(fill="x", pady=4)
        tk.Label(p_row2, text="控制点数量:").pack(side="left")
        self.ctrl_count_var = tk.IntVar(value=16)
        self.spin_ctrl = tk.Spinbox(
            p_row2, from_=4, to=30, textvariable=self.ctrl_count_var, width=6
        )
        self.spin_ctrl.pack(side="left", padx=5)
        tk.Button(
            p_row2, text="重新均匀分布控制点", command=self.re_fit_current_segment
        ).pack(side="right", padx=2)

    def populate_segments_list(self):
        labels = []
        for s in self.app.engine.segments:
            labels.append(f"#{s.segment_id} [{s.event}] ({s.region})")
        self.seg_combobox["values"] = labels
        self.seg_combobox.set("（未选中任何段 - 请点击选择）")
        self.app.select_segment(None)

    def sync_segment_ui(self, seg_idx: Optional[int]):
        if seg_idx is not None and 0 <= seg_idx < len(self.app.engine.segments):
            seg = self.app.engine.segments[seg_idx]
            self.mode_var.set(seg.mode)
            self.ctrl_count_var.set(len(seg.control_points))

    def on_seg_combo_changed(self, event):
        idx = self.seg_combobox.current()
        if idx >= 0:
            self.app.select_segment(idx)
            self.sync_segment_ui(idx)

    def prev_segment(self):
        if not self.app.engine.segments:
            return
        curr = (
            self.app.engine.active_segment_idx
            if self.app.engine.active_segment_idx is not None
            else 1
        )
        new_idx = max(0, curr - 1)
        self.seg_combobox.current(new_idx)
        self.app.select_segment(new_idx)
        self.sync_segment_ui(new_idx)

    def next_segment(self):
        if not self.app.engine.segments:
            return
        curr = (
            self.app.engine.active_segment_idx
            if self.app.engine.active_segment_idx is not None
            else -1
        )
        new_idx = min(len(self.app.engine.segments) - 1, curr + 1)
        self.seg_combobox.current(new_idx)
        self.app.select_segment(new_idx)
        self.sync_segment_ui(new_idx)

    def deselect_segment(self):
        self.seg_combobox.set("（未选中任何段）")
        self.app.select_segment(None)

    def on_show_all_toggled(self):
        show_all = self.show_all_var.get()
        self.app.set_display_mode(show_all)

    def on_mesh_toggled(self):
        show_mesh = self.show_mesh_var.get()
        self.app.mesh_actor.SetVisibility(show_mesh)
        self.app.render()

    def on_mode_changed(self):
        curr_idx = self.app.engine.active_segment_idx
        if curr_idx is not None and 0 <= curr_idx < len(self.app.engine.segments):
            seg = self.app.engine.segments[curr_idx]
            seg.mode = self.mode_var.get()
            seg.dense_xyz = self.app.engine._eval_surface_spline(
                seg.control_points, mode=seg.mode
            )
            self.app.update_active_segment_visuals()

    def re_fit_current_segment(self):
        curr_idx = self.app.engine.active_segment_idx
        if curr_idx is not None and 0 <= curr_idx < len(self.app.engine.segments):
            seg = self.app.engine.segments[curr_idx]
            if seg.event == "SURFACE_SCAN":
                n_ctrl = self.ctrl_count_var.get()
                seg.control_points = self.app.engine._fit_initial_control_points(
                    seg.dense_xyz, n_ctrl
                )
                seg.dense_xyz = self.app.engine._eval_surface_spline(
                    seg.control_points, mode=seg.mode
                )
                self.app.update_active_segment_visuals()

    def open_csv_dialog(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择要编辑的轨迹 CSV 文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.set_csv(Path(path))

    def set_csv(self, path: Path):
        try:
            segments = load_trajectory_file(path)
            self.app.engine.initialize_segments(segments)
            self.populate_segments_list()
            self.app.update_all_visuals()
            self.app._reset_camera()
            print(f"成功载入 {len(self.app.engine.segments)} 条轨迹段！")
        except Exception as e:
            print(f"载入 CSV 失败: {e}")
            messagebox.showerror("载入失败", str(e), parent=self.root)

    def export_dialog(self):
        if not self.app.engine.segments:
            messagebox.showerror("错误", "当前没有轨迹可导出！", parent=self.root)
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出修改后的轨迹 CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
        )
        if not path:
            return

        csv_path = Path(path)
        json_path = csv_path.with_suffix(".json")
        paq_path = csv_path.with_name(csv_path.stem + "_PAQ.txt")

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "segment_id",
                    "point_id",
                    "event",
                    "spray_on",
                    "region",
                    "x",
                    "y",
                    "z",
                    "note",
                ]
            )
            for seg in self.app.engine.segments:
                for pt_id, xyz in enumerate(seg.dense_xyz):
                    writer.writerow(
                        [
                            seg.segment_id,
                            pt_id,
                            seg.event,
                            int(seg.spray_on),
                            seg.region,
                            f"{xyz[0]:.6f}",
                            f"{xyz[1]:.6f}",
                            f"{xyz[2]:.6f}",
                            seg.note + " [Edited]",
                        ]
                    )

        control_payload = []
        for seg in self.app.engine.segments:
            control_payload.append(
                {
                    "segment_id": seg.segment_id,
                    "event": seg.event,
                    "region": seg.region,
                    "points": seg.control_points.tolist(),
                }
            )
        json_path.write_text(
            json.dumps(
                {"control_points": control_payload}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

        all_pts, is_spray = [], []
        for seg in self.app.engine.segments:
            for xyz in seg.dense_xyz:
                all_pts.append(xyz)
                is_spray.append(seg.event == "SURFACE_SCAN")

        if all_pts:
            pts_arr = np.asarray(all_pts, dtype=float)
            closest, _, face_ids = trimesh.proximity.closest_point(
                self.app.mesh, pts_arr
            )
            normals = np.asarray(self.app.mesh.face_normals, dtype=float)[face_ids]
            inward_normals = -normals

            export_pts = pts_arr.copy()
            for i, sp in enumerate(is_spray):
                if sp:
                    export_pts[i] = closest[i] + normals[i] * 0.0

            tangents = np.empty_like(export_pts)
            if len(export_pts) > 1:
                tangents[0] = export_pts[1] - export_pts[0]
                tangents[-1] = export_pts[-1] - export_pts[-2]
                tangents[1:-1] = export_pts[2:] - export_pts[:-2]
                for i in range(len(tangents)):
                    if tangents[i, 2] < 0:
                        tangents[i] = -tangents[i]

            with paq_path.open("w", encoding="utf-8", newline="\n") as f:
                cum_d = 0.0
                for i in range(len(export_pts)):
                    if i > 0:
                        cum_d += float(
                            np.linalg.norm(export_pts[i] - export_pts[i - 1])
                        )
                    quat = _orientation_quaternion(tangents[i], inward_normals[i])
                    f.write(
                        f"{export_pts[i, 0]:.3f} {export_pts[i, 1]:.3f} {export_pts[i, 2]:.3f} "
                        f"{quat[0]:.3f} {quat[1]:.3f} {quat[2]:.3f} {quat[3]:.3f} "
                        f"300.000 0.000 {cum_d:.6f} 0 0.000\n"
                    )

        messagebox.showinfo(
            "完成",
            f"已成功导出所有格式轨迹：\nCSV: {csv_path}\nJSON: {json_path}\nPAQ: {paq_path}",
            parent=self.root,
        )


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


def _orientation_quaternion(
    tangent: np.ndarray, inward_normal: np.ndarray
) -> np.ndarray:
    y_axis = np.asarray(tangent, dtype=float)
    y_norm = np.linalg.norm(y_axis)
    y_axis = y_axis / y_norm if y_norm > EPS else np.array([0.0, 1.0, 0.0])

    z_axis = np.asarray(inward_normal, dtype=float)
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm if z_norm > EPS else np.array([0.0, 0.0, 1.0])

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= EPS:
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, z_axis)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(helper, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis /= max(x_norm, EPS)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), EPS)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return _quaternion_from_matrix(rot_matrix)


def tick_tkinter(panel: TkControlPanel):
    try:
        if panel and panel.root and panel.root.winfo_exists():
            panel.root.update_idletasks()
            panel.root.update()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="OptCuts 3D Surface Trajectory Editor")
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ mesh file.")
    parser.add_argument("--csv", default="", help="Input trajectory CSV file.")
    args = parser.parse_args()

    # 1. 独立置顶对话框选择 STL 文件
    mesh_path = args.mesh
    if not mesh_path:
        temp_tk = tk.Tk()
        temp_tk.withdraw()
        temp_tk.attributes("-topmost", True)
        temp_tk.update()
        mesh_path = filedialog.askopenfilename(
            parent=temp_tk,
            title="【第 1 步】请选择工件 STL/OBJ 三维模型",
            filetypes=[
                ("3D Mesh files", "*.stl *.obj *.ply *.off"),
                ("All files", "*.*"),
            ],
        )
        temp_tk.destroy()

    if not mesh_path:
        print("未选择模型，程序退出。")
        return

    print(f"正在加载工件模型: {mesh_path}")
    mesh_raw = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_raw, trimesh.Scene):
        geoms = [
            g for g in mesh_raw.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = mesh_raw
    mesh.remove_unreferenced_vertices()

    # 2. 检查是否有同名 CSV 文件
    stl_path = Path(mesh_path)
    auto_csv = stl_path.with_name(stl_path.stem + "_optcuts_trajectory.csv")

    # 3. 启动 GUI
    tk_root = tk.Tk()
    app = SurfaceCurveApp(mesh)
    panel = TkControlPanel(app, tk_root)
    app.panel = panel

    # 4. 自动载入同名 CSV 或通过参数载入
    target_csv = (
        Path(args.csv) if args.csv else (auto_csv if auto_csv.exists() else None)
    )
    if target_csv and target_csv.exists():
        print(f"自动载入轨迹文件: {target_csv}")
        panel.set_csv(target_csv)

    app.interactor.AddObserver("TimerEvent", lambda obj, ev: tick_tkinter(panel))
    app.interactor.CreateRepeatingTimer(10)

    app.render_window.Render()
    app.interactor.Initialize()
    app.interactor.Start()


if __name__ == "__main__":
    main()
