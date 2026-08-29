#!/usr/bin/env python3
"""PyQt5 + VTK user interface for spray trajectory and thickness simulation."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

try:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    import vtk
    import vtkmodules.qt
    vtkmodules.qt.PyQtImpl = "PyQt5"
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError as exc:  # Give a useful error if launched from another Python.
    raise SystemExit(
        "This UI requires PyQt5 and VTK. Run it with the xyjpytorch environment, "
        "for example:\n"
        "E:\\AI-DL\\Anoconda2020\\envs\\xyjpytorch\\python.exe "
        "spray_simulation_ui.py"
    ) from exc

try:
    from .trajectorygenerator import (
        TrajectoryParameters,
        TrajectoryResult,
        generate_trajectory,
        load_input_points,
        save_paq,
        save_trajectory_csv,
    )
    from .thicknesspredict import (
        ThicknessParameters,
        ThicknessResult,
        predict_thickness,
        save_colored_ply,
        save_thickness_csv,
    )
except ImportError:
    from trajectorygenerator import (
        TrajectoryParameters,
        TrajectoryResult,
        generate_trajectory,
        load_input_points,
        save_paq,
        save_trajectory_csv,
    )
    from thicknesspredict import (
        ThicknessParameters,
        ThicknessResult,
        predict_thickness,
        save_colored_ply,
        save_thickness_csv,
    )


class CalculationWorker(QThread):
    """Run a Python callable away from the GUI thread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, operation: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.succeeded.emit(self.operation())
        except Exception:
            self.failed.emit(traceback.format_exc())


def make_points_actor(
    points: np.ndarray,
    color=(0.72, 0.76, 0.82),
    point_size: float = 3.0,
    scalars: Optional[np.ndarray] = None,
):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    for point in points:
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    vertices = vtk.vtkCellArray()
    for index in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(index)
    polydata.SetVerts(vertices)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    lookup = None
    if scalars is None:
        mapper.ScalarVisibilityOff()
    else:
        values = vtk.vtkFloatArray()
        values.SetName("Thickness")
        for value in scalars:
            values.InsertNextValue(float(value))
        polydata.GetPointData().SetScalars(values)
        lower = float(np.min(scalars)) if len(scalars) else 0.0
        upper = float(np.max(scalars)) if len(scalars) else 1.0
        if upper <= lower:
            upper = lower + 1.0
        lookup = vtk.vtkLookupTable()
        lookup.SetNumberOfTableValues(256)
        lookup.SetHueRange(0.667, 0.0)
        lookup.SetRange(lower, upper)
        lookup.Build()
        mapper.SetLookupTable(lookup)
        mapper.SetScalarRange(lower, upper)
        mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    if scalars is None:
        actor.GetProperty().SetColor(*color)
    return actor, lookup


def make_segment_actor(
    points: np.ndarray,
    segment_slices,
    color,
    line_width: float,
    selected_types: Optional[np.ndarray] = None,
    desired_type: Optional[str] = None,
):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()
    for point in points:
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    lines = vtk.vtkCellArray()
    for segment in segment_slices:
        if desired_type is not None and selected_types is not None:
            if selected_types[segment.start] != desired_type:
                continue
        count = segment.stop - segment.start
        if count < 2:
            continue
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(count)
        for local_index, point_index in enumerate(range(segment.start, segment.stop)):
            line.GetPointIds().SetId(local_index, point_index)
        lines.InsertNextCell(line)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(line_width)
    return actor


def make_normal_actor(
    points: np.ndarray,
    normals: np.ndarray,
    scale: float,
    maximum: int = 350,
):
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        points = points[indices]
        normals = normals[indices]
    vtk_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for index, (point, normal) in enumerate(zip(points, normals)):
        endpoint = point + scale * normal
        vtk_points.InsertNextPoint(*[float(value) for value in point])
        vtk_points.InsertNextPoint(*[float(value) for value in endpoint])
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 2 * index)
        line.GetPointIds().SetId(1, 2 * index + 1)
        lines.InsertNextCell(line)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.55, 0.05)
    actor.GetProperty().SetLineWidth(1.2)
    return actor


class SpraySimulationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("三维喷涂轨迹与厚度仿真")
        self.resize(1450, 900)
        self.input_path: Optional[Path] = None
        self.preview_points: Optional[np.ndarray] = None
        self.trajectory_result: Optional[TrajectoryResult] = None
        self.thickness_result: Optional[ThicknessResult] = None
        self.worker: Optional[CalculationWorker] = None
        self.actors: Dict[str, object] = {}
        self.scalar_bar = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(430)
        scroll.setWidget(controls)
        root_layout.addWidget(scroll)

        vtk_container = QWidget()
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(vtk_container)
        vtk_layout.addWidget(self.vtk_widget)
        root_layout.addWidget(vtk_container, 1)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.10, 0.12, 0.16)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()
        self._set_action_state()

    @staticmethod
    def _double_box(value, minimum, maximum, decimals=3, step=1.0):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _build_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("数据")
        input_layout = QVBoxLayout(input_group)
        self.input_label = QLabel("尚未导入点云")
        self.input_label.setWordWrap(True)
        self.import_button = QPushButton("导入点云 / STL")
        self.import_button.clicked.connect(self.import_cloud)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.import_button)
        layout.addWidget(input_group)

        trajectory_group = QGroupBox("轨迹参数")
        trajectory_form = QFormLayout(trajectory_group)
        self.trajectory_spacing = self._double_box(0.0, 0.0, 1.0e6)
        self.trajectory_spacing.setSpecialValueText("自动")
        self.point_spacing = self._double_box(10.0, 0.001, 1.0e6)
        self.spray_distance = self._double_box(50.0, 0.0, 1.0e6)
        self.speed = self._double_box(50.0, 0.001, 1.0e6)
        self.max_points = QSpinBox()
        self.max_points.setRange(100, 1000000)
        self.max_points.setValue(4000)
        self.normal_neighbors = QSpinBox()
        self.normal_neighbors.setRange(3, 500)
        self.normal_neighbors.setValue(24)
        trajectory_form.addRow("覆盖轨迹间距", self.trajectory_spacing)
        trajectory_form.addRow("轨迹点间距", self.point_spacing)
        trajectory_form.addRow("喷涂距离", self.spray_distance)
        trajectory_form.addRow("速度", self.speed)
        trajectory_form.addRow("最大点云数", self.max_points)
        trajectory_form.addRow("法向邻域点数", self.normal_neighbors)
        self.calculate_trajectory_button = QPushButton("开始轨迹计算")
        self.calculate_trajectory_button.clicked.connect(self.calculate_trajectory)
        self.flip_button = QPushButton("反转法向量")
        self.flip_button.clicked.connect(self.flip_normals)
        trajectory_form.addRow(self.calculate_trajectory_button)
        trajectory_form.addRow(self.flip_button)
        layout.addWidget(trajectory_group)

        thickness_group = QGroupBox("厚度模型参数")
        thickness_form = QFormLayout(thickness_group)
        self.gamma_max = self._double_box(1.0, 0.0, 1.0e9, 6, 0.1)
        self.h_opt = self._double_box(50.0, 0.001, 1.0e6)
        self.sigma_h = self._double_box(10.0, 0.001, 1.0e6)
        self.angle_k = self._double_box(1.0, 0.0, 100.0)
        self.constant_c = self._double_box(1.0, 0.0, 1.0e12, 6, 0.1)
        self.alpha = self._double_box(30.0, 0.01, 89.0)
        self.beta = self._double_box(1.0, -0.99, 100.0)
        self.include_transitions = QCheckBox("过渡段也参与喷涂")
        thickness_form.addRow("γmax", self.gamma_max)
        thickness_form.addRow("hopt", self.h_opt)
        thickness_form.addRow("σh", self.sigma_h)
        thickness_form.addRow("角度指数 k", self.angle_k)
        thickness_form.addRow("常数 C", self.constant_c)
        thickness_form.addRow("喷锥半角 α (°)", self.alpha)
        thickness_form.addRow("β", self.beta)
        thickness_form.addRow(self.include_transitions)
        self.calculate_thickness_button = QPushButton("开始厚度仿真")
        self.calculate_thickness_button.clicked.connect(self.calculate_thickness)
        thickness_form.addRow(self.calculate_thickness_button)
        layout.addWidget(thickness_group)

        visibility_group = QGroupBox("显示控制")
        visibility_layout = QVBoxLayout(visibility_group)
        self.visibility_checks: Dict[str, QCheckBox] = {}
        labels = {
            "cloud": "显示点云",
            "surface": "显示表面轨迹",
            "nozzle": "显示喷枪轨迹",
            "normals": "显示法向量",
            "thickness": "显示厚度",
        }
        for key, text in labels.items():
            check = QCheckBox(text)
            check.setChecked(key != "normals")
            check.toggled.connect(lambda visible, name=key: self.set_actor_visibility(name, visible))
            self.visibility_checks[key] = check
            visibility_layout.addWidget(check)
        layout.addWidget(visibility_group)

        clear_group = QGroupBox("清除结果")
        clear_layout = QVBoxLayout(clear_group)
        self.clear_cloud_button = QPushButton("清除点云")
        self.clear_cloud_button.clicked.connect(self.clear_cloud)
        self.clear_trajectory_button = QPushButton("清除轨迹")
        self.clear_trajectory_button.clicked.connect(self.clear_trajectory)
        self.clear_thickness_button = QPushButton("清除厚度结果")
        self.clear_thickness_button.clicked.connect(self.clear_thickness)
        self.clear_all_button = QPushButton("全部清除")
        self.clear_all_button.clicked.connect(self.clear_all)
        clear_layout.addWidget(self.clear_cloud_button)
        clear_layout.addWidget(self.clear_trajectory_button)
        clear_layout.addWidget(self.clear_thickness_button)
        clear_layout.addWidget(self.clear_all_button)
        layout.addWidget(clear_group)

        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout(export_group)
        self.export_trajectory_button = QPushButton("导出轨迹 CSV + PAQ")
        self.export_trajectory_button.clicked.connect(self.export_trajectory)
        self.export_thickness_button = QPushButton("导出厚度 CSV + PLY")
        self.export_thickness_button.clicked.connect(self.export_thickness)
        export_layout.addWidget(self.export_trajectory_button)
        export_layout.addWidget(self.export_thickness_button)
        layout.addWidget(export_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        return widget

    def trajectory_parameters(self) -> TrajectoryParameters:
        return TrajectoryParameters(
            max_points=self.max_points.value(),
            normal_neighbors=self.normal_neighbors.value(),
            trajectory_spacing=self.trajectory_spacing.value(),
            point_spacing=self.point_spacing.value(),
            spray_distance=self.spray_distance.value(),
            speed=self.speed.value(),
        )

    def thickness_parameters(self) -> ThicknessParameters:
        return ThicknessParameters(
            gamma_max=self.gamma_max.value(),
            spray_height=self.spray_distance.value(),
            h_opt=self.h_opt.value(),
            sigma_h=self.sigma_h.value(),
            angle_power_k=self.angle_k.value(),
            constant_c=self.constant_c.value(),
            cone_half_angle_degrees=self.alpha.value(),
            beta=self.beta.value(),
            include_transitions=self.include_transitions.isChecked(),
        )

    def import_cloud(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择点云或STL", "", "支持文件 (*.ply *.stl);;PLY点云 (*.ply);;STL模型 (*.stl)"
        )
        if not filename:
            return
        try:
            self.input_path = Path(filename)
            self.preview_points, _ = load_input_points(
                self.input_path, self.trajectory_parameters()
            )
            self.trajectory_result = None
            self.thickness_result = None
            self.input_label.setText(str(self.input_path))
            self._clear_scene()
            # Importing a new cloud is an explicit request to view it.  Do
            # not let a stale unchecked visibility box create a hidden actor.
            self.visibility_checks["cloud"].setChecked(True)
            actor, _ = make_points_actor(self.preview_points)
            self._replace_actor("cloud", actor)
            self.renderer.ResetCamera()
            self.render()
            self.status_label.setText(f"已导入 {len(self.preview_points)} 个点")
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
        self._set_action_state()

    def _start_worker(
        self,
        operation: Callable[[], object],
        success: Callable[[object], None],
        message: str,
    ) -> None:
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self._set_busy(True)
        worker = CalculationWorker(operation, self)
        worker.succeeded.connect(success)
        worker.failed.connect(self._calculation_failed)
        worker.finished.connect(self._worker_finished)
        self.worker = worker
        worker.start()

    def calculate_trajectory(self) -> None:
        if self.input_path is None:
            return
        path = self.input_path
        parameters = self.trajectory_parameters()
        self._start_worker(
            lambda: generate_trajectory(path, parameters),
            self._trajectory_finished,
            "正在计算三维轨迹……",
        )

    def _trajectory_finished(self, result: object) -> None:
        self.trajectory_result = result  # type: ignore[assignment]
        self.thickness_result = None
        self._update_trajectory_actors(reset_camera=True)
        trajectory = self.trajectory_result
        self.status_label.setText(
            f"轨迹完成：{trajectory.segment_count} 段，"
            f"{trajectory.coverage_count} 条喷涂轨迹，{len(trajectory.surface_points)} 个点"
        )

    def flip_normals(self) -> None:
        if self.trajectory_result is None:
            return
        self.trajectory_result = self.trajectory_result.flipped_normals()
        self.thickness_result = None
        self._remove_actor("thickness")
        self._update_trajectory_actors(reset_camera=False)
        self.status_label.setText("法向量已整体反转，喷枪轨迹和姿态已同步更新")
        self._set_action_state()

    def calculate_thickness(self) -> None:
        if self.trajectory_result is None:
            return
        trajectory = self.trajectory_result
        parameters = self.thickness_parameters()
        self._start_worker(
            lambda: predict_thickness(
                trajectory.cloud_points,
                trajectory.cloud_normals,
                trajectory,
                parameters,
            ),
            self._thickness_finished,
            "正在计算厚度贡献……",
        )

    def _thickness_finished(self, result: object) -> None:
        self.thickness_result = result  # type: ignore[assignment]
        actor, lookup = make_points_actor(
            self.thickness_result.cloud_points,
            point_size=5.0,
            scalars=self.thickness_result.thickness,
        )
        self._replace_actor("thickness", actor)
        if self.scalar_bar is not None:
            self.renderer.RemoveActor2D(self.scalar_bar)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(lookup)
        self.scalar_bar.SetTitle("Thickness")
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.SetVisibility(self.visibility_checks["thickness"].isChecked())
        self.renderer.AddActor2D(self.scalar_bar)
        self.render()
        result_value = self.thickness_result
        self.status_label.setText(
            f"厚度完成：min={result_value.minimum:.5g}，max={result_value.maximum:.5g}，"
            f"mean={result_value.mean:.5g}，均匀度={result_value.uniformity:.3f}"
        )

    def _update_trajectory_actors(self, reset_camera: bool) -> None:
        result = self.trajectory_result
        if result is None:
            return
        cloud_actor, _ = make_points_actor(result.cloud_points)
        surface_actor = make_segment_actor(
            result.surface_points,
            result.segment_slices,
            (0.15, 0.95, 0.35),
            2.5,
        )
        nozzle_actor = make_segment_actor(
            result.nozzle_points,
            result.segment_slices,
            (0.95, 0.15, 0.25),
            3.0,
        )
        extent = float(np.linalg.norm(np.ptp(result.cloud_points, axis=0)))
        normal_actor = make_normal_actor(
            result.surface_points,
            result.surface_normals,
            max(0.025 * extent, 0.25 * self.spray_distance.value()),
        )
        self._replace_actor("cloud", cloud_actor)
        self._replace_actor("surface", surface_actor)
        self._replace_actor("nozzle", nozzle_actor)
        self._replace_actor("normals", normal_actor)
        self._remove_actor("thickness")
        if reset_camera:
            self.renderer.ResetCamera()
        self.render()

    def _replace_actor(self, name: str, actor) -> None:
        self._remove_actor(name)
        self.actors[name] = actor
        actor.SetVisibility(self.visibility_checks[name].isChecked())
        self.renderer.AddActor(actor)

    def _remove_actor(self, name: str) -> None:
        actor = self.actors.pop(name, None)
        if actor is not None:
            self.renderer.RemoveActor(actor)
        if name == "thickness" and self.scalar_bar is not None:
            self.renderer.RemoveActor2D(self.scalar_bar)
            self.scalar_bar = None

    def _clear_scene(self) -> None:
        for name in list(self.actors):
            self._remove_actor(name)

    def _discard_thickness(self) -> None:
        self.thickness_result = None
        self._remove_actor("thickness")

    def _discard_trajectory(self) -> None:
        self._discard_thickness()
        self.trajectory_result = None
        for name in ("surface", "nozzle", "normals"):
            self._remove_actor(name)

    def _discard_cloud(self) -> None:
        self._discard_trajectory()
        self.preview_points = None
        self.input_path = None
        self._remove_actor("cloud")
        self.input_label.setText("尚未导入点云")

    def clear_thickness(self) -> None:
        """Remove thickness data, heat-map actor and scalar bar."""
        self._discard_thickness()
        self.progress_bar.setValue(0)
        self.status_label.setText("已清除厚度结果")
        self.render()
        self._set_action_state()

    def clear_trajectory(self) -> None:
        """Remove surface/nozzle/normal actors and dependent thickness data."""
        self._discard_trajectory()
        self.progress_bar.setValue(0)
        self.status_label.setText("已清除轨迹及其厚度结果")
        self.render()
        self._set_action_state()

    def clear_cloud(self) -> None:
        """Remove the cloud and every result derived from it."""
        self._discard_cloud()
        self.progress_bar.setValue(0)
        self.status_label.setText("已清除点云、轨迹和厚度结果")
        self.renderer.ResetCamera()
        self.render()
        self._set_action_state()

    def clear_all(self) -> None:
        """Reset all imported and calculated data in the current session."""
        self._discard_cloud()
        # Restore the same visibility defaults as a fresh application.  Actor
        # visibility otherwise survives clearing and can hide the next import.
        for name, check in self.visibility_checks.items():
            check.setChecked(name != "normals")
        self.progress_bar.setValue(0)
        self.status_label.setText("已全部清除")
        self.renderer.ResetCamera()
        self.render()
        self._set_action_state()

    def set_actor_visibility(self, name: str, visible: bool) -> None:
        actor = self.actors.get(name)
        if actor is not None:
            actor.SetVisibility(visible)
        if name == "thickness" and self.scalar_bar is not None:
            self.scalar_bar.SetVisibility(visible)
        self.render()

    def export_trajectory(self) -> None:
        if self.trajectory_result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存轨迹", "segmented_spray_trajectory.csv", "CSV文件 (*.csv)"
        )
        if not filename:
            return
        csv_path = Path(filename).with_suffix(".csv")
        paq_path = csv_path.with_suffix(".txt")
        try:
            save_trajectory_csv(csv_path, self.trajectory_result)
            save_paq(paq_path, self.trajectory_result)
            self.status_label.setText(f"已导出：{csv_path.name} 和 {paq_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_thickness(self) -> None:
        if self.thickness_result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存厚度", "thickness_result.csv", "CSV文件 (*.csv)"
        )
        if not filename:
            return
        csv_path = Path(filename).with_suffix(".csv")
        ply_path = csv_path.with_suffix(".ply")
        try:
            save_thickness_csv(csv_path, self.thickness_result)
            save_colored_ply(ply_path, self.thickness_result)
            self.status_label.setText(f"已导出：{csv_path.name} 和 {ply_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _calculation_failed(self, details: str) -> None:
        self.status_label.setText("计算失败")
        QMessageBox.critical(self, "计算失败", details)

    def _worker_finished(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if "失败" not in self.status_label.text() else 0)
        self.worker = None
        self._set_busy(False)
        self._set_action_state()

    def _set_busy(self, busy: bool) -> None:
        self.import_button.setEnabled(not busy)
        self.calculate_trajectory_button.setEnabled(not busy and self.input_path is not None)
        self.calculate_thickness_button.setEnabled(not busy and self.trajectory_result is not None)
        self.flip_button.setEnabled(not busy and self.trajectory_result is not None)
        self.clear_cloud_button.setEnabled(not busy and self.input_path is not None)
        self.clear_trajectory_button.setEnabled(not busy and self.trajectory_result is not None)
        self.clear_thickness_button.setEnabled(not busy and self.thickness_result is not None)
        self.clear_all_button.setEnabled(not busy and bool(self.actors))

    def _set_action_state(self) -> None:
        busy = self.worker is not None and self.worker.isRunning()
        self.calculate_trajectory_button.setEnabled(not busy and self.input_path is not None)
        self.calculate_thickness_button.setEnabled(not busy and self.trajectory_result is not None)
        self.flip_button.setEnabled(not busy and self.trajectory_result is not None)
        self.export_trajectory_button.setEnabled(self.trajectory_result is not None)
        self.export_thickness_button.setEnabled(self.thickness_result is not None)
        self.clear_cloud_button.setEnabled(not busy and self.input_path is not None)
        self.clear_trajectory_button.setEnabled(
            not busy and (
                self.trajectory_result is not None
                or any(name in self.actors for name in ("surface", "nozzle", "normals"))
            )
        )
        self.clear_thickness_button.setEnabled(
            not busy and (self.thickness_result is not None or "thickness" in self.actors)
        )
        self.clear_all_button.setEnabled(
            not busy and (
                self.input_path is not None
                or self.trajectory_result is not None
                or self.thickness_result is not None
                or bool(self.actors)
            )
        )

    def render(self) -> None:
        if hasattr(self, "vtk_widget"):
            self.vtk_widget.GetRenderWindow().Render()

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "计算进行中", "请等待当前计算完成后再关闭窗口。")
            event.ignore()
            return
        self.vtk_widget.Finalize()
        super().closeEvent(event)


def main() -> None:
    application = QApplication(sys.argv)
    window = SpraySimulationWindow()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
