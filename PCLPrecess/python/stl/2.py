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
#手动调整代码

# =========================================================================
# 一、 轨迹数据结构与高容错文件解析
# =========================================================================

@dataclass
class TrackSegmentData:
    segment_id: int
    event: str
    spray_on: bool
    region: str
    note: str
    dense_xyz: np.ndarray  # 当前曲面插值/拟合点阵 (K, 3)
    control_points: np.ndarray  # 位于曲面上的控制点 (N, 3)
    mode: str = "SPLINE"  # "SPLINE" 或 "LINE"


def _as_points(value: object, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{label} 必须包含有效的三维坐标。")
    return points


def load_trajectory_file(csv_path: Path, metadata_path: Optional[Path] = None) -> List[TrackSegmentData]:
    """从 CSV/JSON 中读取完整轨迹段落（高容错机制）。"""
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

            segments_raw.setdefault(seg_id, []).append((pt_id, event, spray_on, region, pt, note))

    control_by_segment: Dict[int, np.ndarray] = {}
    json_path = metadata_path if metadata_path else csv_path.with_suffix(".json")
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for item in meta.get("control_points", []):
                control_by_segment[int(item["segment_id"])] = _as_points(item["points"], "control_points")
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

        segments.append(TrackSegmentData(
            segment_id=seg_id,
            event=event,
            spray_on=spray_on,
            region=region,
            note=note,
            dense_xyz=pts,
            control_points=ctrls,
        ))

    return segments


# =========================================================================
# 二、 多段流形拟合与过渡缝合计算核心 (防崩溃强化)
# =========================================================================

class MultiSegmentSplineEngine:
    def __init__(self, mesh: trimesh.Trimesh, sample_step: float = 1.0):
        self.mesh = mesh
        self.sample_step = max(float(sample_step), 0.2)
        self.segments: List[TrackSegmentData] = []
        self.active_segment_idx: int = 0
        self.show_all_segments: bool = True

    def initialize_segments(self, seg_list: List[TrackSegmentData], default_ctrl_count: int = 10):
        self.segments = seg_list
        for seg in self.segments:
            if seg.event == "SURFACE_SCAN" and len(seg.dense_xyz) >= 2:
                if len(seg.control_points) < 4 or len(seg.control_points) == len(seg.dense_xyz):
                    seg.control_points = self._fit_initial_control_points(seg.dense_xyz, default_ctrl_count)
                seg.dense_xyz = self._eval_surface_spline(seg.control_points, mode=seg.mode)
            else:
                seg.control_points = np.vstack([seg.dense_xyz[0], seg.dense_xyz[-1]]) if len(
                    seg.dense_xyz) >= 2 else seg.dense_xyz.copy()

        self._rebuild_all_transitions()

    def _clean_points(self, pts: np.ndarray) -> np.ndarray:
        """去除连续重合点，防止底层 Fortran/C 数组除零崩溃。"""
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.r_[True, dists > 1e-4]
        cleaned = pts[keep]
        return cleaned if len(cleaned) >= 2 else pts

    def _fit_initial_control_points(self, polyline: np.ndarray, num_ctrl: int) -> np.ndarray:
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

    def _eval_surface_spline(self, ctrl_pts: np.ndarray, mode: str = "SPLINE") -> np.ndarray:
        pts = self._clean_points(ctrl_pts)
        n = len(pts)
        if n < 2:
            return pts.copy()

        if mode == "LINE" or n < 3:
            return self._eval_linear_surface(pts)

        # 构建严格单调递增的参数节点序列
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dists = np.maximum(dists, 1e-4)
        cum = np.r_[0.0, np.cumsum(dists)]
        total = max(cum[-1], 1e-5)
        u_knots = cum / total

        # 严格强制单调递增
        for i in range(1, len(u_knots)):
            if u_knots[i] <= u_knots[i - 1]:
                u_knots[i] = u_knots[i - 1] + 1e-4
        u_knots /= u_knots[-1]

        k_deg = min(3, n - 1)
        try:
            tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1], pts[:, 2]], u=u_knots, k=k_deg, s=0.0)
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

    def _c1_connector(self, p_start: np.ndarray, p_end: np.ndarray,
                      t_start: np.ndarray, t_end: np.ndarray) -> np.ndarray:
        gap = float(np.linalg.norm(p_end - p_start))
        if gap <= 1.0e-7:
            return np.vstack([p_start, p_end])

        t0 = t_start / max(float(np.linalg.norm(t_start)), 1.0e-9)
        t1 = t_end / max(float(np.linalg.norm(t_end)), 1.0e-9)
        handle = 0.45 * gap
        n_samples = max(4, int(np.ceil(gap / self.sample_step)) + 1)
        u = np.linspace(0.0, 1.0, n_samples)[:, None]

        curve = ((2 * u ** 3 - 3 * u ** 2 + 1) * p_start + (u ** 3 - 2 * u ** 2 + u) * handle * t0 +
                 (-2 * u ** 3 + 3 * u ** 2) * p_end + (u ** 3 - u ** 2) * handle * t1)
        return curve

    def _rebuild_neighbor_transitions(self, seg_idx: int):
        if seg_idx > 0 and self.segments[seg_idx - 1].event in {"OVERTRAVEL", "SEAM_JUMP"}:
            prev_seg = self.segments[seg_idx - 1]
            curr_seg = self.segments[seg_idx]
            p0 = prev_seg.dense_xyz[0]
            p1 = curr_seg.dense_xyz[0]
            t0 = (prev_seg.dense_xyz[1] - prev_seg.dense_xyz[0]) if len(prev_seg.dense_xyz) >= 2 else (p1 - p0)
            t1 = (curr_seg.dense_xyz[1] - curr_seg.dense_xyz[0]) if len(curr_seg.dense_xyz) >= 2 else (p1 - p0)
            prev_seg.dense_xyz = self._c1_connector(p0, p1, t0, t1)
            prev_seg.control_points = np.vstack([prev_seg.dense_xyz[0], prev_seg.dense_xyz[-1]])

        if seg_idx < len(self.segments) - 1 and self.segments[seg_idx + 1].event in {"OVERTRAVEL", "SEAM_JUMP"}:
            next_seg = self.segments[seg_idx + 1]
            curr_seg = self.segments[seg_idx]
            p0 = curr_seg.dense_xyz[-1]
            p1 = next_seg.dense_xyz[-1]
            t0 = (curr_seg.dense_xyz[-1] - curr_seg.dense_xyz[-2]) if len(curr_seg.dense_xyz) >= 2 else (p1 - p0)
            t1 = (next_seg.dense_xyz[-1] - next_seg.dense_xyz[-2]) if len(next_seg.dense_xyz) >= 2 else (p1 - p0)
            next_seg.dense_xyz = self._c1_connector(p0, p1, t0, t1)
            next_seg.control_points = np.vstack([next_seg.dense_xyz[0], next_seg.dense_xyz[-1]])

    def _rebuild_all_transitions(self):
        for i in range(len(self.segments)):
            if self.segments[i].event == "SURFACE_SCAN":
                self._rebuild_neighbor_transitions(i)


# =========================================================================
# 三、 VTK 交互式表面拖拽器
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
            actor.GetProperty().SetColor(1.0, 0.95, 0.0)
            self.app.render()
            return

        self.OnLeftButtonDown()

    def on_mouse_move(self, obj, event):
        if self.is_dragging and self.drag_ctrl_idx is not None:
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
# 四、 VTK 视口与可视化渲染主程序
# =========================================================================

class SurfaceCurveApp:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.engine = MultiSegmentSplineEngine(mesh)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.14, 0.18)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetWindowName("3D Mesh Trajectory Designer - [Surface Spline & Solo Mode]")
        self.render_window.SetSize(1450, 900)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.mesh_actor = self._create_mesh_actor(mesh)
        self.renderer.AddActor(self.mesh_actor)

        self.track_actors: Dict[int, vtk.vtkActor] = {}
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

        self.refresh_handles()
        self._update_text()
        self.render()

    def update_active_segment_visuals(self):
        curr_idx = self.engine.active_segment_idx
        self._create_or_update_track_actor(curr_idx)
        if curr_idx > 0:
            self._create_or_update_track_actor(curr_idx - 1)
        if curr_idx < len(self.engine.segments) - 1:
            self._create_or_update_track_actor(curr_idx + 1)

        seg = self.engine.segments[curr_idx]
        for actor, c_i in list(self.handle_actor_map.items()):
            if c_i < len(seg.control_points):
                pt = seg.control_points[c_i]
                actor.SetPosition(float(pt[0]), float(pt[1]), float(pt[2]))

        self._update_text()
        self.render()

    def _create_or_update_track_actor(self, seg_idx: int):
        seg = self.engine.segments[seg_idx]
        pts = seg.dense_xyz
        if len(pts) < 2:
            return

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

        if seg_idx in self.track_actors:
            actor = self.track_actors[seg_idx]
            actor.SetMapper(mapper)
        else:
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRenderLinesAsTubes(True)
            self.renderer.AddActor(actor)
            self.track_actors[seg_idx] = actor

        is_active = (seg_idx == self.engine.active_segment_idx)

        # 全显示 vs 仅显示选中段
        if not self.engine.show_all_segments:
            actor.SetVisibility(is_active)
        else:
            actor.SetVisibility(True)

        if is_active and seg.event == "SURFACE_SCAN":
            actor.GetProperty().SetColor(0.0, 1.0, 0.4)
            actor.GetProperty().SetLineWidth(4.5)
        elif is_active:
            actor.GetProperty().SetColor(1.0, 0.9, 0.2)
            actor.GetProperty().SetLineWidth(4.0)
        elif seg.event == "SURFACE_SCAN":
            actor.GetProperty().SetColor(0.88, 0.12, 0.12)
            actor.GetProperty().SetLineWidth(3.0)
        else:
            actor.GetProperty().SetColor(0.1, 0.8, 0.2)
            actor.GetProperty().SetLineWidth(2.0)

    def refresh_handles(self):
        for actor in self.handle_actors:
            self.renderer.RemoveActor(actor)
        self.handle_actors.clear()
        self.handle_actor_map.clear()

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

    def select_segment(self, seg_idx: int):
        if 0 <= seg_idx < len(self.engine.segments):
            self.engine.active_segment_idx = seg_idx
            self.update_all_visuals()

    def set_display_mode(self, show_all: bool):
        self.engine.show_all_segments = show_all
        for seg_idx, actor in self.track_actors.items():
            if not show_all:
                actor.SetVisibility(seg_idx == self.engine.active_segment_idx)
            else:
                actor.SetVisibility(True)
        self.render()

    def _update_text(self):
        if not self.engine.segments:
            self.info_text_actor.SetInput("请在控制台加载轨迹 CSV 文件")
            return

        seg = self.engine.segments[self.engine.active_segment_idx]
        mode_name = "五次/三次 B 样条 (Spline)" if seg.mode == "SPLINE" else "曲面直线段 (Line)"
        disp_name = "【显示全部轨迹】" if self.engine.show_all_segments else "【仅显示选中段 (Solo)】"
        info = (
            f"当前活动段: Segment #{seg.segment_id} [{seg.event} - {seg.region}] | 总段数: {len(self.engine.segments)}\n"
            f"显示模式: {disp_name} | 曲线算法: {mode_name}\n"
            f"控制点数量: {len(seg.control_points)} | 采样点数: {len(seg.dense_xyz)}\n"
            f"操作提示: 鼠标左键直接拖动控制球即可贴体变形；在控制面板中可快速切换段落或显示模式。"
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
# 五、 独立 Tkinter 控制面板 UI
# =========================================================================

class TkControlPanel:
    def __init__(self, app: SurfaceCurveApp, tk_root: tk.Tk):
        self.app = app
        self.root = tk_root
        self.root.title("轨迹分段编辑与显示控制台")
        self.root.geometry("420x520+50+50")
        self.root.wm_attributes("-topmost", True)
        self.root.deiconify()

        # 1. 文件区
        f_frame = tk.LabelFrame(self.root, text="文件载入与导出", padx=8, pady=6)
        f_frame.pack(fill="x", padx=10, pady=5)

        btn_row = tk.Frame(f_frame)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="打开轨迹 CSV", command=self.open_csv_dialog,
                  bg="#1948CD", fg="white", font=("Arial", 9, "bold")).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_row, text="保存导出 (CSV/PAQ)", command=self.export_dialog,
                  bg="#008040", fg="white", font=("Arial", 9, "bold")).pack(side="right", expand=True, fill="x", padx=2)

        # 2. 轨迹段选择器
        seg_frame = tk.LabelFrame(self.root, text="轨迹段切换 (Segment Selector)", padx=8, pady=6)
        seg_frame.pack(fill="x", padx=10, pady=5)

        combo_row = tk.Frame(seg_frame)
        combo_row.pack(fill="x", pady=2)
        tk.Label(combo_row, text="目标轨迹段:").pack(side="left")
        self.combo_var = tk.StringVar()
        self.seg_combobox = ttk.Combobox(combo_row, textvariable=self.combo_var, state="readonly")
        self.seg_combobox.pack(side="right", expand=True, fill="x", padx=5)
        self.seg_combobox.bind("<<ComboboxSelected>>", self.on_seg_combo_changed)

        nav_row = tk.Frame(seg_frame)
        nav_row.pack(fill="x", pady=4)
        tk.Button(nav_row, text="◀ 上一段 (Prev)", command=self.prev_segment).pack(side="left", expand=True, fill="x",
                                                                                   padx=2)
        tk.Button(nav_row, text="下一段 (Next) ▶", command=self.next_segment).pack(side="right", expand=True, fill="x",
                                                                                   padx=2)

        # 3. 显示与过滤模式
        disp_frame = tk.LabelFrame(self.root, text="显示与过滤模式 (Display & Filter)", padx=8, pady=6)
        disp_frame.pack(fill="x", padx=10, pady=5)

        self.show_all_var = tk.BooleanVar(value=True)
        self.cb_show_all = tk.Checkbutton(
            disp_frame, text="全显示模式 (勾选: 显示全部; 取消: 仅显示当前段)",
            variable=self.show_all_var, command=self.on_show_all_toggled, font=("Arial", 9, "bold")
        )
        self.cb_show_all.pack(anchor="w", pady=2)

        self.show_mesh_var = tk.BooleanVar(value=True)
        self.cb_mesh = tk.Checkbutton(
            disp_frame, text="显示 3D 网格模型 (Mesh)",
            variable=self.show_mesh_var, command=self.on_mesh_toggled
        )
        self.cb_mesh.pack(anchor="w", pady=2)

        # 4. 当前段样条参数调节
        param_frame = tk.LabelFrame(self.root, text="当前段样条参数", padx=8, pady=6)
        param_frame.pack(fill="x", padx=10, pady=5)

        p_row1 = tk.Frame(param_frame)
        p_row1.pack(fill="x", pady=2)
        tk.Label(p_row1, text="绘制模式:").pack(side="left")
        self.mode_var = tk.StringVar(value="SPLINE")
        tk.Radiobutton(p_row1, text="五次/三次 B 样条", variable=self.mode_var,
                       value="SPLINE", command=self.on_mode_changed).pack(side="left", padx=5)
        tk.Radiobutton(p_row1, text="直线段", variable=self.mode_var,
                       value="LINE", command=self.on_mode_changed).pack(side="left", padx=5)

        p_row2 = tk.Frame(param_frame)
        p_row2.pack(fill="x", pady=4)
        tk.Label(p_row2, text="控制点数量:").pack(side="left")
        self.ctrl_count_var = tk.IntVar(value=10)
        self.spin_ctrl = tk.Spinbox(p_row2, from_=4, to=30, textvariable=self.ctrl_count_var, width=6)
        self.spin_ctrl.pack(side="left", padx=5)
        tk.Button(p_row2, text="重新均匀分布控制点", command=self.re_fit_current_segment).pack(side="right", padx=2)

    def populate_segments_list(self):
        labels = []
        for s in self.app.engine.segments:
            labels.append(f"#{s.segment_id} [{s.event}] ({s.region})")
        self.seg_combobox["values"] = labels
        if labels:
            idx = min(self.app.engine.active_segment_idx, len(labels) - 1)
            self.seg_combobox.current(idx)
            self.sync_segment_ui(idx)

    def sync_segment_ui(self, seg_idx: int):
        if 0 <= seg_idx < len(self.app.engine.segments):
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
        new_idx = max(0, self.app.engine.active_segment_idx - 1)
        self.seg_combobox.current(new_idx)
        self.app.select_segment(new_idx)
        self.sync_segment_ui(new_idx)

    def next_segment(self):
        if not self.app.engine.segments:
            return
        new_idx = min(len(self.app.engine.segments) - 1, self.app.engine.active_segment_idx + 1)
        self.seg_combobox.current(new_idx)
        self.app.select_segment(new_idx)
        self.sync_segment_ui(new_idx)

    def on_show_all_toggled(self):
        show_all = self.show_all_var.get()
        self.app.set_display_mode(show_all)

    def on_mesh_toggled(self):
        show_mesh = self.show_mesh_var.get()
        self.app.mesh_actor.SetVisibility(show_mesh)
        self.app.render()

    def on_mode_changed(self):
        curr_idx = self.app.engine.active_segment_idx
        if 0 <= curr_idx < len(self.app.engine.segments):
            seg = self.app.engine.segments[curr_idx]
            seg.mode = self.mode_var.get()
            seg.dense_xyz = self.app.engine._eval_surface_spline(seg.control_points, mode=seg.mode)
            self.app.update_active_segment_visuals()

    def re_fit_current_segment(self):
        curr_idx = self.app.engine.active_segment_idx
        if 0 <= curr_idx < len(self.app.engine.segments):
            seg = self.app.engine.segments[curr_idx]
            if seg.event == "SURFACE_SCAN":
                n_ctrl = self.ctrl_count_var.get()
                seg.control_points = self.app.engine._fit_initial_control_points(seg.dense_xyz, n_ctrl)
                seg.dense_xyz = self.app.engine._eval_surface_spline(seg.control_points, mode=seg.mode)
                self.app.update_all_visuals()

    def open_csv_dialog(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择要编辑的轨迹 CSV 文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            try:
                segments = load_trajectory_file(Path(path))
                self.app.engine.initialize_segments(segments)
                self.populate_segments_list()
                self.app.update_all_visuals()
                self.app._reset_camera()
                messagebox.showinfo("成功", f"成功载入并拟合 {len(segments)} 条轨迹段！", parent=self.root)
            except Exception as e:
                messagebox.showerror("载入失败", str(e), parent=self.root)

    def export_dialog(self):
        if not self.app.engine.segments:
            messagebox.showerror("错误", "当前没有轨迹可导出！", parent=self.root)
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出修改后的轨迹 CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not path:
            return

        csv_path = Path(path)
        json_path = csv_path.with_suffix(".json")
        paq_path = csv_path.with_name(csv_path.stem + "_PAQ.txt")

        # 1. 导出 CSV
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["segment_id", "point_id", "event", "spray_on", "region", "x", "y", "z", "note"])
            for seg in self.app.engine.segments:
                for pt_id, xyz in enumerate(seg.dense_xyz):
                    writer.writerow([
                        seg.segment_id, pt_id, seg.event, int(seg.spray_on), seg.region,
                        f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}",
                        seg.note + " [Edited]"
                    ])

        # 2. 导出 JSON
        control_payload = []
        for seg in self.app.engine.segments:
            control_payload.append({
                "segment_id": seg.segment_id,
                "event": seg.event,
                "region": seg.region,
                "points": seg.control_points.tolist(),
            })
        json_path.write_text(json.dumps({"control_points": control_payload}, indent=2, ensure_ascii=False),
                             encoding="utf-8")

        # 3. 导出 PAQ 文件
        all_pts, is_spray = [], []
        for seg in self.app.engine.segments:
            for xyz in seg.dense_xyz:
                all_pts.append(xyz)
                is_spray.append(seg.event == "SURFACE_SCAN")

        if all_pts:
            pts_arr = np.asarray(all_pts, dtype=float)
            closest, _, face_ids = trimesh.proximity.closest_point(self.app.mesh, pts_arr)
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
                        cum_d += float(np.linalg.norm(export_pts[i] - export_pts[i - 1]))
                    quat = _orientation_quaternion(tangents[i], inward_normals[i])
                    f.write(
                        f"{export_pts[i, 0]:.3f} {export_pts[i, 1]:.3f} {export_pts[i, 2]:.3f} "
                        f"{quat[0]:.3f} {quat[1]:.3f} {quat[2]:.3f} {quat[3]:.3f} "
                        f"300.000 0.000 {cum_d:.6f} 0 0.000\n"
                    )

        messagebox.showinfo("完成", f"已成功导出所有格式轨迹：\nCSV: {csv_path}\nJSON: {json_path}\nPAQ: {paq_path}",
                            parent=self.root)


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


# =========================================================================
# 六、 启动与交互流程
# =========================================================================

def _select_mesh_file() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="【第 1 步】请选择工件 STL/OBJ 三维模型",
        filetypes=[("3D Mesh files", "*.stl *.obj *.ply *.off"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


def _select_csv_file() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="【第 2 步】请选择要导入编辑的轨迹 CSV 文件 (*_optcuts_trajectory.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


def main():
    parser = argparse.ArgumentParser(description="OptCuts 3D Surface Trajectory Editor")
    parser.add_argument("mesh", nargs="?", default="", help="Input STL/OBJ mesh file.")
    parser.add_argument("--csv", default="", help="Input trajectory CSV file.")
    args = parser.parse_args()

    # 初始化单个全局 Tk 实例
    tk_root = tk.Tk()
    tk_root.withdraw()

    # 1. 弹出第 1 步：选择 STL 模型
    mesh_path = args.mesh
    if not mesh_path:
        mesh_path = filedialog.askopenfilename(
            parent=tk_root,
            title="【第 1 步】请选择工件 STL/OBJ 三维模型",
            filetypes=[("3D Mesh files", "*.stl *.obj *.ply *.off"), ("All files", "*.*")]
        )

    if not mesh_path:
        print("未选择模型，程序退出。")
        tk_root.destroy()
        return

    print(f"正在加载工件模型: {mesh_path}")
    mesh_raw = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_raw, trimesh.Scene):
        geoms = [g for g in mesh_raw.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = mesh_raw
    mesh.remove_unreferenced_vertices()

    # 2. 弹出第 2 步：选择轨迹 CSV 文件
    csv_path = args.csv
    if not csv_path:
        csv_path = filedialog.askopenfilename(
            parent=tk_root,
            title="【第 2 步】请选择要导入编辑的轨迹 CSV 文件 (*_optcuts_trajectory.csv)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

    # 3. 启动应用程序与控制器
    app = SurfaceCurveApp(mesh)
    panel = TkControlPanel(app, tk_root)
    app.panel = panel

    # 4. 自动载入轨迹并生成分段样条与控制点
    if csv_path:
        try:
            segments = load_trajectory_file(Path(csv_path))
            app.engine.initialize_segments(segments)
            panel.populate_segments_list()
            app.update_all_visuals()
            app._reset_camera()
            print(f"成功载入 {len(segments)} 条轨迹段！")
        except Exception as e:
            print(f"载入 CSV 失败: {e}")
            messagebox.showerror("载入失败", str(e), parent=tk_root)

    # 5. 注册定时器与启动渲染循环
    app.interactor.AddObserver("TimerEvent", lambda obj, ev: tick_tkinter(panel))
    app.interactor.CreateRepeatingTimer(10)

    app.render_window.Render()
    app.interactor.Initialize()
    app.interactor.Start()


if __name__ == "__main__":
    main()