"""Fast VTK/OpenGL rendering for OptCuts meshes and trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def _vtk():
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    except ImportError as exc:
        raise RuntimeError("VTK is required for display; install it with `pip install vtk`.") from exc
    return vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray


def _polydata(points, cells=None, cell_kind="lines"):
    vtk, numpy_to_vtk, numpy_to_ids = _vtk()
    data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    values = np.ascontiguousarray(points, dtype=np.float64)
    vtk_points.SetData(numpy_to_vtk(values, deep=True))
    data.SetPoints(vtk_points)
    if cells is not None and len(cells):
        cells = np.ascontiguousarray(cells, dtype=np.int64)
        width = cells.shape[1]
        records = np.column_stack((np.full(len(cells), width), cells)).ravel()
        array = vtk.vtkCellArray()
        array.SetCells(len(cells), numpy_to_ids(records, deep=True))
        {"lines": data.SetLines, "polys": data.SetPolys, "verts": data.SetVerts}[cell_kind](array)
    return data


def _triangles(points, faces):
    return _polydata(points, np.asarray(faces, dtype=np.int64), "polys")


def _lines(items: Sequence[np.ndarray], dimensions: int):
    vtk, numpy_to_vtk, numpy_to_ids = _vtk()
    items = [np.asarray(item, dtype=float) for item in items if len(item) >= 2]
    data = vtk.vtkPolyData()
    if not items:
        return data
    if dimensions == 2:
        items = [np.column_stack((item[:, :2], np.full(len(item), 0.002))) for item in items]
    points = np.vstack(items)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    data.SetPoints(vtk_points)
    records, offset = [], 0
    for item in items:
        records.append(np.r_[len(item), np.arange(offset, offset + len(item))])
        offset += len(item)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(items), numpy_to_ids(np.ascontiguousarray(np.concatenate(records), dtype=np.int64), deep=True))
    data.SetLines(cells)
    return data


def _points(values, dimensions: int):
    values = np.asarray(values, dtype=float).reshape((-1, dimensions))
    if dimensions == 2:
        values = np.column_stack((values, np.full(len(values), 0.004)))
    cells = np.arange(len(values), dtype=np.int64).reshape((-1, 1))
    return _polydata(values, cells, "verts")


def _actor(data, color, size=1.0, opacity=1.0, points=False):
    vtk, _, _ = _vtk()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(size)
    if points:
        actor.GetProperty().SetPointSize(size)
        actor.GetProperty().RenderPointsAsSpheresOn()
    return actor


def _title(renderer, text):
    vtk, _, _ = _vtk()
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.GetTextProperty().SetFontSize(20)
    actor.GetTextProperty().SetColor(0.08, 0.08, 0.08)
    actor.GetTextProperty().SetJustificationToCentered()
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.5, 0.955)
    renderer.AddActor2D(actor)
    return actor


def _legend(renderer, entries):
    if not entries:
        return
    vtk, _, _ = _vtk()
    legend = vtk.vtkLegendBoxActor()
    legend.SetNumberOfEntries(len(entries))
    for index, (actor, label, color) in enumerate(entries):
        # vtkLegendBoxActor expects representative PolyData, not the actor.
        legend.SetEntry(index, actor.GetMapper().GetInput(), label, color)
    legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
    legend.SetPosition(0.63, 0.70)
    legend.SetPosition2(0.35, min(0.26, 0.045 * len(entries) + 0.04))
    legend.UseBackgroundOn()
    legend.SetBackgroundColor(1.0, 1.0, 1.0)
    legend.SetBackgroundOpacity(0.82)
    renderer.AddActor2D(legend)


def _window(left, right, name, size=(1800, 950), lower_right=None):
    vtk, _, _ = _vtk()
    left.SetViewport(0.0, 0.0, 0.5, 1.0)
    if lower_right is None:
        right.SetViewport(0.5, 0.0, 1.0, 1.0)
        renderers = (left, right)
    else:
        right.SetViewport(0.5, 0.38, 1.0, 1.0)
        lower_right.SetViewport(0.5, 0.0, 1.0, 0.37)
        renderers = (left, right, lower_right)
    for renderer in renderers:
        renderer.SetBackground(0.97, 0.97, 0.97)
        renderer.UseFXAAOn()
    window = vtk.vtkRenderWindow()
    window.SetWindowName(name)
    window.SetSize(*size)
    window.SetMultiSamples(8)
    for renderer in renderers:
        window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    return window, interactor


def _add_visibility_controls(
    window,
    interactor,
    left_renderer,
    right_renderer,
    left_model,
    left_trajectory,
    right_model,
    right_trajectory,
):
    """Add four clickable overlay options for independent visibility control."""
    vtk, _, _ = _vtk()
    controls = []

    def add_label(renderer, text, y, actors):
        label = vtk.vtkTextActor()
        label.SetInput(f"[ ] {text}")
        label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        label.SetPosition(0.025, y)
        prop = label.GetTextProperty()
        prop.SetFontSize(18)
        prop.SetColor(0.08, 0.08, 0.08)
        prop.SetBackgroundColor(1.0, 1.0, 1.0)
        prop.SetBackgroundOpacity(0.82)
        prop.SetFrame(True)
        prop.SetFrameColor(0.35, 0.35, 0.35)
        renderer.AddActor2D(label)
        controls.append({
            "label": label,
            "text": text,
            "y": y,
            "actors": list(actors),
            "renderer": renderer,
            "hidden": False,
        })

    add_label(left_renderer, "Hide model", 0.925, left_model)
    if left_trajectory:
        add_label(left_renderer, "Hide trajectory", 0.880, left_trajectory)
    add_label(right_renderer, "Hide model", 0.925, right_model)
    if right_trajectory:
        add_label(right_renderer, "Hide trajectory", 0.880, right_trajectory)

    def on_left_button(caller, _event):
        x, y = caller.GetEventPosition()
        width, height = window.GetSize()
        for control in controls:
            vx0, vy0, vx1, vy1 = control["renderer"].GetViewport()
            viewport_width = (vx1 - vx0) * width
            viewport_height = (vy1 - vy0) * height
            x0 = vx0 * width + 0.018 * viewport_width
            x1 = vx0 * width + 0.23 * viewport_width
            y0 = vy0 * height + (control["y"] - 0.008) * viewport_height
            y1 = vy0 * height + (control["y"] + 0.032) * viewport_height
            if x0 <= x <= x1 and y0 <= y <= y1:
                control["hidden"] = not control["hidden"]
                visible = not control["hidden"]
                for actor in control["actors"]:
                    actor.SetVisibility(visible)
                mark = "x" if control["hidden"] else " "
                control["label"].SetInput(f"[{mark}] {control['text']}")
                window.Render()
                return

    interactor.AddObserver("LeftButtonPressEvent", on_left_button, 1.0)
    # Keep Python callbacks/actors alive for the lifetime of the interactor.
    interactor._visibility_controls = controls
    interactor._visibility_callback = on_left_button


def request_spacing(mesh, uv, uv_faces, initial_spacing, cut_edges=None, planning_frame=None,
                    initial_speed=300.0, initial_spray_distance=0.0,
                    initial_a_max=1000.0, initial_j_max=5000.0,
                    initial_arc_lock_guard_d=2.0, initial_arc_speed_loss_pct=0.0):
    """Preview the mapped result and collect all trajectory parameters."""
    vtk, _, _ = _vtk()
    view3d, view2d = vtk.vtkRenderer(), vtk.vtkRenderer()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    surface = _actor(_triangles(vertices, faces), (0.68, 0.78, 0.88), opacity=0.68)
    surface.GetProperty().EdgeVisibilityOn()
    surface.GetProperty().SetEdgeColor(0.30, 0.34, 0.37)
    view3d.AddActor(surface)

    uv3 = np.column_stack((uv, np.zeros(len(uv))))
    chart = _actor(_triangles(uv3, uv_faces), (0.12, 0.12, 0.12), size=0.7)
    chart.GetProperty().SetRepresentationToWireframe()
    view2d.AddActor(chart)
    if cut_edges is not None and len(cut_edges):
        view3d.AddActor(_actor(_lines(cut_edges, 3), (0.86, 0.08, 0.24), 3.0))
    if planning_frame is not None:
        reference = np.vstack((
            planning_frame.origin,
            planning_frame.origin + planning_frame.waist_width * planning_frame.x_axis,
        ))
        view2d.AddActor(_actor(_lines([reference], 2), (0.09, 0.75, 0.81), 3.0))

    _title(view3d, "3D OptCuts result")
    _title(view2d, "2D OptCuts UV result")
    window, interactor = _window(view3d, view2d, "OptCuts spacing setup - VTK/OpenGL")
    view3d.ResetCamera();
    view2d.ResetCamera()
    view2d.GetActiveCamera().SetParallelProjection(True)

    state = {
        "d": {"text": f"{float(initial_spacing):g}", "editing": False, "error": ""},
        "speed": {"text": f"{float(initial_speed):g}", "editing": False, "error": ""},
        "spray_distance": {"text": f"{float(initial_spray_distance):g}", "editing": False, "error": ""},
        "a_max": {"text": f"{float(initial_a_max):g}", "editing": False, "error": ""},
        "j_max": {"text": f"{float(initial_j_max):g}", "editing": False, "error": ""},
        "arc_lock_guard_d": {"text": f"{float(initial_arc_lock_guard_d):g}", "editing": False, "error": ""},
        "arc_speed_loss_pct": {"text": f"{float(initial_arc_speed_loss_pct):g}", "editing": False, "error": ""},
        "active": "d",
        "accepted": None,
    }

    def create_input_actor(y_pos):
        actor = vtk.vtkTextActor()
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(0.03, y_pos)
        prop = actor.GetTextProperty()
        prop.SetFontSize(20)
        prop.SetColor(0.08, 0.08, 0.08)
        prop.SetBackgroundColor(1.0, 1.0, 1.0)
        prop.SetBackgroundOpacity(0.92)
        prop.SetFrame(True)
        prop.SetFrameColor(0.35, 0.35, 0.35)
        view2d.AddActor2D(actor)
        return actor

    input_d = create_input_actor(0.27)
    input_speed = create_input_actor(0.225)
    input_spray = create_input_actor(0.18)
    input_a = create_input_actor(0.135)
    input_j = create_input_actor(0.09)
    input_guard = create_input_actor(0.045)
    input_loss = create_input_actor(0.0)

    button = vtk.vtkTextActor()
    button.SetInput("Generate")
    button.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    button.SetPosition(0.34, 0.29)
    button.GetTextProperty().SetFontSize(22)
    button.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    button.GetTextProperty().SetBackgroundColor(0.10, 0.45, 0.75)
    button.GetTextProperty().SetBackgroundOpacity(1.0)
    button.GetTextProperty().SetFrame(True)
    view2d.AddActor2D(button)

    labels = {
        "d": "d: ", "speed": "speed: ", "spray_distance": "spray_distance: ",
        "a_max": "a_max: ", "j_max": "j_max: ",
        "arc_lock_guard_d": "arc_guard: ", "arc_speed_loss_pct": "arc_loss_%: ",
    }

    def refresh():
        actors = [
            ("d", input_d), ("speed", input_speed), ("spray_distance", input_spray),
            ("a_max", input_a), ("j_max", input_j),
            ("arc_lock_guard_d", input_guard), ("arc_speed_loss_pct", input_loss),
        ]
        for key, actor in actors:
            s = state[key]
            suffix = f"  [{s['error']}]" if s["error"] else ""
            is_active = (state["active"] == key)
            frame_color = (0.0, 0.5, 1.0) if is_active else (0.35, 0.35, 0.35)
            bg_color = (0.9, 0.95, 1.0) if is_active else (1.0, 1.0, 1.0)

            actor.SetInput(f"{labels[key]}{s['text']}_{suffix}")
            actor.GetTextProperty().SetFrameColor(*frame_color)
            actor.GetTextProperty().SetBackgroundColor(*bg_color)
        window.Render()

    def accept():
        errors = {key: "" for key in labels}
        vals = {}
        for key in labels:
            try:
                val = float(state[key]["text"])
                minimum = 0.0 if key in ("spray_distance", "arc_lock_guard_d", "arc_speed_loss_pct") else 0.0
                if not np.isfinite(val) or val < minimum:
                    raise ValueError("Must be > 0")
                if key in ("d", "speed", "a_max", "j_max") and val <= 0.0:
                    raise ValueError("Must be > 0")
                if key == "arc_speed_loss_pct" and val > 100.0:
                    raise ValueError("Must be <= 100")
                vals[key] = val
            except ValueError:
                errors[key] = "Invalid"

        if any(errors.values()):
            for key in labels:
                state[key]["error"] = errors[key]
            refresh()
            return

        state["accepted"] = vals
        interactor.TerminateApp()

    def on_key(caller, _event):
        key = caller.GetKeySym()
        char = caller.GetKeyCode() or ""
        if not char or char not in "0123456789.-+eE":
            candidate = key[3:] if key.startswith("KP_") else key
            char = candidate if len(candidate) == 1 else ""

        active_key = state["active"]
        state[active_key]["error"] = ""

        if key in ("Return", "KP_Enter"):
            accept();
            return
        if key == "Tab":
            order = list(labels)
            idx = order.index(active_key)
            state["active"] = order[(idx + 1) % len(order)]
            refresh();
            return

        if key in ("BackSpace", "Delete"):
            state[active_key]["editing"] = True
            state[active_key]["text"] = state[active_key]["text"][:-1]
        elif char and char in "0123456789.-+eE":
            if not state[active_key]["editing"]:
                state[active_key]["text"] = ""
                state[active_key]["editing"] = True
            state[active_key]["text"] += char
        refresh()

    def on_click(caller, _event):
        x, y = caller.GetEventPosition()
        width, height = window.GetSize()

        def check_click(y_pos, key):
            vx0, vy0, vx1, vy1 = view2d.GetViewport()
            viewport_height = (vy1 - vy0) * height
            viewport_width = (vx1 - vx0) * width
            y0 = vy0 * height + (y_pos - 0.01) * viewport_height
            y1 = vy0 * height + (y_pos + 0.04) * viewport_height
            x0 = vx0 * width + 0.02 * viewport_width
            x1 = vx0 * width + 0.32 * viewport_width
            if x0 <= x <= x1 and y0 <= y <= y1:
                state["active"] = key
                state[key]["editing"] = True
                return True
            return False

        input_positions = [
            (0.27, "d"), (0.225, "speed"), (0.18, "spray_distance"),
            (0.135, "a_max"), (0.09, "j_max"), (0.045, "arc_lock_guard_d"),
            (0.0, "arc_speed_loss_pct"),
        ]
        if any(check_click(y_pos, key) for y_pos, key in input_positions):
            refresh();
            return

        if 0.5 * width + 0.33 * 0.5 * width <= x <= 0.5 * width + 0.49 * 0.5 * width and y <= 0.12 * height:
            accept()

    interactor.AddObserver("KeyPressEvent", on_key, 1.0)
    interactor.AddObserver("LeftButtonPressEvent", on_click, 1.0)
    interactor._spacing_callbacks = (on_key, on_click, input_d, input_speed, input_spray,
                                     input_a, input_j, input_guard, input_loss, button)
    refresh()
    interactor.Initialize();
    interactor.Start()
    window.Finalize()
    return state["accepted"]

def _add_optimization_chart(renderer, metadata):
    """Add the d versus penalty curve and return the retained VTK objects."""
    vtk, numpy_to_vtk, _ = _vtk()
    curve = metadata.get("optimization_curve", {})
    spacing = np.asarray(curve.get("spacing", []), dtype=float)
    objective = np.asarray(curve.get("objective", []), dtype=float)
    if not len(spacing) or spacing.shape != objective.shape:
        return None

    table = vtk.vtkTable()
    x_values = numpy_to_vtk(np.ascontiguousarray(spacing), deep=True)
    x_values.SetName("d")
    y_values = numpy_to_vtk(np.ascontiguousarray(objective), deep=True)
    y_values.SetName("F(d)")
    table.AddColumn(x_values)
    table.AddColumn(y_values)

    chart = vtk.vtkChartXY()
    chart.SetTitle("Spacing penalty")
    chart.GetAxis(vtk.vtkAxis.BOTTOM).SetTitle("d")
    chart.GetAxis(vtk.vtkAxis.LEFT).SetTitle("F(d)")
    line = chart.AddPlot(vtk.vtkChart.LINE)
    line.SetInputData(table, 0, 1)
    line.SetColor(24, 105, 170, 255)
    line.SetWidth(2.0)

    best = int(np.argmin(objective))
    best_table = vtk.vtkTable()
    best_x = numpy_to_vtk(np.asarray([spacing[best]], dtype=float), deep=True)
    best_x.SetName("d*")
    best_y = numpy_to_vtk(np.asarray([objective[best]], dtype=float), deep=True)
    best_y.SetName("F(d*)")
    best_table.AddColumn(best_x)
    best_table.AddColumn(best_y)
    point = chart.AddPlot(vtk.vtkChart.POINTS)
    point.SetInputData(best_table, 0, 1)
    point.SetColor(210, 45, 45, 255)
    point.SetMarkerStyle(vtk.vtkPlotPoints.CIRCLE)
    point.SetMarkerSize(8.0)

    context = vtk.vtkContextActor()
    context.GetScene().AddItem(chart)
    context.GetScene().SetRenderer(renderer)
    renderer.AddActor(context)
    renderer._optimization_chart = (context, chart, table, best_table)
    return renderer._optimization_chart


def _add_actual_spacing_text(renderer, metadata):
    values = np.asarray(metadata.get("optimization_actual_spacings", []), dtype=float)
    if values.shape != (3,):
        return None
    vtk, _, _ = _vtk()
    actor = vtk.vtkTextActor()
    actor.SetInput(
        "Actual spacing   "
        f"center: {values[0]:.6g}   upper: {values[1]:.6g}   lower: {values[2]:.6g}"
    )
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.5, 0.015)
    prop = actor.GetTextProperty()
    prop.SetFontSize(18)
    prop.SetColor(0.08, 0.08, 0.08)
    prop.SetJustificationToCentered()
    prop.SetBackgroundColor(1.0, 1.0, 1.0)
    prop.SetBackgroundOpacity(0.86)
    renderer.AddActor2D(actor)
    return actor


def show_result(
    mesh,
    uv,
    uv_faces,
    cut_edges=None,
    trajectory=None,
    planning_frame=None,
    metadata=None,
):
    """Render the final 3D and UV views as batched GPU geometry."""
    vtk, _, _ = _vtk()
    view3d, view2d = vtk.vtkRenderer(), vtk.vtkRenderer()
    metadata = metadata or {}
    penalty_view = vtk.vtkRenderer() if metadata.get("optimization_curve") else None
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    surface = _actor(_triangles(vertices, faces), (0.68, 0.78, 0.88), opacity=0.68)
    surface.GetProperty().EdgeVisibilityOn()
    surface.GetProperty().SetEdgeColor(0.30, 0.34, 0.37)
    surface.GetProperty().SetLineWidth(0.6)
    view3d.AddActor(surface)
    uv3 = np.column_stack((uv, np.zeros(len(uv))))
    chart = _actor(_triangles(uv3, uv_faces), (0.12, 0.12, 0.12), size=0.7)
    chart.GetProperty().SetRepresentationToWireframe()
    view2d.AddActor(chart)
    model3d, trajectory3d = [surface], []
    model2d, trajectory2d = [chart], []

    legends3d, legends2d = [], []
    if cut_edges is not None and len(cut_edges):
        actor = _actor(_lines(cut_edges, 3), (0.86, 0.08, 0.24), 3.0)
        view3d.AddActor(actor)
        model3d.append(actor)
        legends3d.append((actor, f"OptCuts seams ({len(cut_edges)})", (0.86, 0.08, 0.24)))
    trajectory = trajectory or []
    colors = {
        "SURFACE_SCAN": (0.84, 0.15, 0.16),
        "SEAM_JUMP": (0.12, 0.47, 0.71),
        "SEAM_EDGE_CONNECT": (0.12, 0.47, 0.71),
        "SEAM_BLEND_3D": (1.00, 0.50, 0.05),
        "OVERTRAVEL": (0.17, 0.63, 0.17),
    }
    for kind, color in colors.items():
        paths = [item.xyz for item in trajectory if item.kind == kind and len(item.xyz) >= 2]
        if paths:
            actor = _actor(_lines(paths, 3), color, 2.3 if kind == "SURFACE_SCAN" else 1.6)
            view3d.AddActor(actor)
            trajectory3d.append(actor)
            legends3d.append((actor, kind, color))
    scans = [item.uv for item in trajectory if item.kind == "SURFACE_SCAN" and len(item.uv) >= 2]
    if scans:
        actor = _actor(_lines(scans, 2), colors["SURFACE_SCAN"], 2.0)
        view2d.AddActor(actor)
        trajectory2d.append(actor)
        legends2d.append((actor, "surface scan", colors["SURFACE_SCAN"]))
    edge_connectors = [
        item.uv for item in trajectory
        if item.kind == "SEAM_EDGE_CONNECT" and len(item.uv) >= 2
    ]
    if edge_connectors:
        actor = _actor(_lines(edge_connectors, 2), colors["SEAM_EDGE_CONNECT"], 1.6)
        view2d.AddActor(actor)
        trajectory2d.append(actor)
        legends2d.append((actor, "nearest seam edge connect", colors["SEAM_EDGE_CONNECT"]))
    jumps = [point for item in trajectory if item.kind == "SEAM_JUMP" for point in item.uv]
    if jumps:
        actor = _actor(_points(jumps, 2), colors["SEAM_JUMP"], 7.0, points=True)
        view2d.AddActor(actor)
        trajectory2d.append(actor)
        legends2d.append((actor, "SEAM_JUMP pair", colors["SEAM_JUMP"]))
    surfaces = [item for item in trajectory if item.kind == "SURFACE_SCAN" and len(item.xyz) >= 2]
    if surfaces:
        for xyz, uv_point, color, label in (
            (surfaces[0].xyz[0], surfaces[0].uv[0], (0.2, 0.8, 0.2), "start"),
            (surfaces[-1].xyz[-1], surfaces[-1].uv[-1], (1.0, 0.0, 1.0), "end"),
        ):
            a3 = _actor(_points([xyz], 3), color, 11.0, points=True)
            a2 = _actor(_points([uv_point], 2), color, 11.0, points=True)
            view3d.AddActor(a3); view2d.AddActor(a2)
            trajectory3d.append(a3); trajectory2d.append(a2)
            legends3d.append((a3, label, color)); legends2d.append((a2, label, color))
    if planning_frame is not None:
        reference = np.vstack((planning_frame.origin, planning_frame.origin + planning_frame.waist_width * planning_frame.x_axis))
        actor = _actor(_lines([reference], 2), (0.09, 0.75, 0.81), 3.0)
        view2d.AddActor(actor)
        model2d.append(actor)
        legends2d.append((actor, "reference line 1", (0.09, 0.75, 0.81)))
    _title(view3d, "3D seam-aware spray trajectory")
    _title(view2d, "2D OptCuts UV trajectory")
    _legend(view3d, legends3d); _legend(view2d, legends2d)
    spacing_text = _add_actual_spacing_text(view2d, metadata)
    if penalty_view is not None:
        _add_optimization_chart(penalty_view, metadata)
    window, interactor = _window(
        view3d, view2d, "OptCuts trajectory - VTK/OpenGL",
        lower_right=penalty_view,
    )
    _add_visibility_controls(
        window, interactor, view3d, view2d,
        model3d, trajectory3d, model2d, trajectory2d,
    )
    view3d.ResetCamera(); view2d.ResetCamera()
    camera = view2d.GetActiveCamera()
    camera.SetParallelProjection(True)
    camera.SetViewUp(0.0, 1.0, 0.0)
    view2d.ResetCameraClippingRange()
    if spacing_text is not None:
        interactor._actual_spacing_text = spacing_text
    window.Render(); interactor.Initialize(); interactor.Start()


def _image(rgb):
    vtk, numpy_to_vtk, _ = _vtk()
    rgb = np.ascontiguousarray(np.flipud(rgb), dtype=np.uint8)
    height, width = rgb.shape[:2]
    image = vtk.vtkImageData()
    image.SetDimensions(width, height, 1)
    scalars = numpy_to_vtk(rgb.reshape((-1, 3)), deep=True)
    scalars.SetNumberOfComponents(3)
    image.GetPointData().SetScalars(scalars)
    return image


def play_iteration_animation(mesh, result_obj: Path, read_obj: Callable, final_hold_ms: int):
    """Play OptCuts iteration frames through a VTK render-window timer."""
    from PIL import Image
    vtk, numpy_to_vtk, _ = _vtk()
    gif_path, directory = result_obj.with_name("anim.gif"), result_obj.with_name("mesh_frames")
    if not gif_path.exists():
        print(f"OptCuts iteration animation was not found: {gif_path}"); return
    frames, durations = [], []
    with Image.open(gif_path) as gif:
        for index in range(getattr(gif, "n_frames", 1)):
            gif.seek(index); frames.append(np.asarray(gif.convert("RGB")).copy())
            durations.append(max(int(gif.info.get("duration", 100)), 20))
    paths = sorted(directory.glob("frame_*.obj"), key=lambda item: int(item.stem.split("_")[-1]))
    count = min(len(frames), len(paths))
    if not count:
        print(f"OptCuts 3D parameter-grid frames were not found: {directory}"); return
    frames, paths = frames[:count], paths[:count]
    interval = int(np.median(durations[:count])); hold = max(1, round(final_hold_ms / max(interval, 1)))
    playback = list(range(count)) + [count - 1] * hold
    scale = 20.0 / max(float(np.max(np.ptp(np.asarray(mesh.vertices), axis=0))), 1.0e-12)

    def frame_mesh(index):
        xyz, faces, uv, uv_faces = read_obj(paths[index])
        data = _triangles(xyz, faces)
        centers = uv[uv_faces].mean(axis=1)
        checker = ((np.floor(centers[:, 0] * scale) + np.floor(centers[:, 1] * scale)).astype(int) & 1)
        colors = np.where(checker[:, None] == 0, [[20, 20, 20]], [[235, 235, 235]]).astype(np.uint8)
        array = numpy_to_vtk(np.ascontiguousarray(colors), deep=True); array.SetNumberOfComponents(3)
        data.GetCellData().SetScalars(array)
        return data

    view3d, view_image = vtk.vtkRenderer(), vtk.vtkRenderer()
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(frame_mesh(0))
    mapper.SetScalarModeToUseCellData(); mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.25, 0.25, 0.25); view3d.AddActor(actor)
    image_actor = vtk.vtkImageActor(); image_actor.SetInputData(_image(frames[0])); view_image.AddActor(image_actor)
    _title(view3d, "Parameter grid mapped on 3D workpiece")
    title = _title(view_image, f"OptCuts iteration 1/{count} (looping)")
    window, interactor = _window(view3d, view_image, "OptCuts iterations - VTK/OpenGL", (1800, 900))
    view3d.ResetCamera(); view_image.ResetCamera(); state = {"position": 0}

    def update(_caller, _event):
        state["position"] = (state["position"] + 1) % len(playback)
        index = playback[state["position"]]
        mapper.SetInputData(frame_mesh(index)); image_actor.SetInputData(_image(frames[index]))
        title.SetInput(f"OptCuts iteration {index + 1}/{count} (looping)")
        view3d.ResetCameraClippingRange(); window.Render()

    interactor.AddObserver("TimerEvent", update)
    window.Render(); interactor.Initialize(); interactor.CreateRepeatingTimer(max(interval, 16)); interactor.Start()
