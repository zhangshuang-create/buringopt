"""Run the official OptCuts example on an STL selected from a file dialog."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTCUTS_ROOT = PROJECT_ROOT / "third_party" / "OptCuts"
OPTCUTS_EXE = OPTCUTS_ROOT / "build" / "Release" / "OptCuts_bin.exe"


def select_stl_file() -> Path | None:
    """Open a file dialog and return the selected STL path."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Select an STL file for OptCuts",
        filetypes=[("STL files", "*.stl")],
    )
    root.destroy()
    return Path(selected) if selected else None


def convert_stl_to_obj(stl_path: Path) -> Path:
    """Convert STL to OBJ because official OptCuts only accepts OBJ/OFF."""
    mesh = trimesh.load_mesh(stl_path, file_type="stl", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The selected STL does not contain a triangle mesh.")

    # STL stores every triangle independently and has no shared vertex indices.
    # Restore that connectivity before exporting to the topology-based OptCuts.
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    component_count = len(mesh.split(only_watertight=False))
    if component_count != 1:
        raise ValueError(
            "Official OptCuts requires exactly one connected component; "
            f"the selected STL contains {component_count}."
        )

    input_directory = OPTCUTS_ROOT / "python_input"
    input_directory.mkdir(exist_ok=True)
    obj_path = input_directory / "optcuts_input.obj"
    mesh.export(obj_path, file_type="obj")
    return obj_path


def run_official_optcuts(input_obj: Path) -> Path:
    """Run OptCuts with the exact parameters from its official README."""
    if not OPTCUTS_EXE.is_file():
        raise FileNotFoundError(f"OptCuts executable not found: {OPTCUTS_EXE}")

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
        "firstTrial",
    ]
    print("Starting official OptCuts. The optimization window may take time to initialize...", flush=True)
    subprocess.run(command, cwd=OPTCUTS_ROOT, env=environment, check=True)
    print("OptCuts finished. Loading the VTK result window...", flush=True)

    output_root = OPTCUTS_ROOT / "output"
    candidates = [
        path
        for path in output_root.glob("*/finalResult_mesh.obj")
        if path.parent.name.startswith(input_obj.stem + "_")
    ]
    if not candidates:
        raise FileNotFoundError("OptCuts did not produce finalResult_mesh.obj.")

    result_obj = max(candidates, key=lambda path: path.stat().st_mtime)
    normalized_result = result_obj.with_name("finalResult_mesh_normalizedUV.obj")
    return normalized_result if normalized_result.is_file() else result_obj


def show_result_with_vtk(result_obj: Path) -> None:
    """Show the official result as textured 3D geometry and a UV mesh."""
    try:
        import vtk
    except ImportError as exc:
        raise RuntimeError("VTK is required for result visualization.") from exc

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(result_obj))
    reader.Update()
    result_mesh = reader.GetOutput()
    texture_coordinates = result_mesh.GetPointData().GetTCoords()
    if result_mesh.GetNumberOfPoints() == 0:
        raise ValueError(f"VTK could not read the OptCuts result: {result_obj}")
    if texture_coordinates is None:
        raise ValueError("The OptCuts result does not contain UV coordinates.")

    # A cut in the OptCuts output is represented by a boundary edge: after
    # cutting, the edge belongs to only one triangle.  Build a matching line
    # overlay for the geometric and UV views from those single-use edges.
    edge_use_count: dict[tuple[int, int], int] = {}
    polys = result_mesh.GetPolys()
    polys.InitTraversal()
    cell_ids = vtk.vtkIdList()
    while polys.GetNextCell(cell_ids):
        ids = [cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())]
        for index, first in enumerate(ids):
            second = ids[(index + 1) % len(ids)]
            edge = (min(first, second), max(first, second))
            edge_use_count[edge] = edge_use_count.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_use_count.items() if count == 1]

    # Keep only duplicated geometric boundary edges.  The original open
    # border is also a single-use edge, while a cut has two UV copies at the
    # same 3D location.  Rounding makes the comparison robust to OBJ output
    # precision while retaining the original vertex IDs for each overlay.
    def point_key(point: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(round(float(value), 8) for value in point)

    geometric_edge_uvs: dict[
        tuple[tuple[float, float, float], tuple[float, float, float]],
        set[tuple[tuple[float, float], tuple[float, float]]],
    ] = {}
    for first, second in boundary_edges:
        first_xyz = point_key(result_mesh.GetPoint(first))
        second_xyz = point_key(result_mesh.GetPoint(second))
        endpoints = sorted((first_xyz, second_xyz))
        geometric_key = (endpoints[0], endpoints[1])
        first_uv = texture_coordinates.GetTuple2(first)
        second_uv = texture_coordinates.GetTuple2(second)
        uv_endpoints = tuple(sorted((
            tuple(round(float(value), 8) for value in first_uv),
            tuple(round(float(value), 8) for value in second_uv),
        )))
        geometric_edge_uvs.setdefault(geometric_key, set()).add(uv_endpoints)
    cut_edges = []
    for edge in boundary_edges:
        first, second = edge
        endpoints = sorted((point_key(result_mesh.GetPoint(first)), point_key(result_mesh.GetPoint(second))))
        geometric_key = (endpoints[0], endpoints[1])
        # A real cut has coincident 3D edges with distinct UV coordinates.
        if len(geometric_edge_uvs[geometric_key]) > 1:
            cut_edges.append(edge)

    def make_edge_overlay(use_uv_coordinates: bool) -> vtk.vtkPolyData:
        overlay = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(result_mesh.GetNumberOfPoints())
        for point_index in range(result_mesh.GetNumberOfPoints()):
            if use_uv_coordinates:
                u, v = texture_coordinates.GetTuple2(point_index)
                points.SetPoint(point_index, u, v, 0.002)
            else:
                points.SetPoint(point_index, result_mesh.GetPoint(point_index))
        overlay.SetPoints(points)
        lines = vtk.vtkCellArray()
        for first, second in cut_edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, first)
            line.GetPointIds().SetId(1, second)
            lines.InsertNextCell(line)
        overlay.SetLines(lines)
        return overlay

    cut_mapper = vtk.vtkPolyDataMapper()
    cut_mapper.SetInputData(make_edge_overlay(use_uv_coordinates=False))
    cut_mapper.ScalarVisibilityOff()
    cut_actor = vtk.vtkActor()
    cut_actor.SetMapper(cut_mapper)
    cut_actor.GetProperty().SetColor(1.0, 0.85, 0.05)
    cut_actor.GetProperty().SetLineWidth(1.5)

    uv_cut_mapper = vtk.vtkPolyDataMapper()
    uv_cut_mapper.SetInputData(make_edge_overlay(use_uv_coordinates=True))
    uv_cut_mapper.ScalarVisibilityOff()
    uv_cut_actor = vtk.vtkActor()
    uv_cut_actor.SetMapper(uv_cut_mapper)
    uv_cut_actor.GetProperty().SetColor(1.0, 0.35, 0.0)
    uv_cut_actor.GetProperty().SetLineWidth(1.5)

    checker_source = vtk.vtkImageCanvasSource2D()
    checker_source.SetScalarTypeToUnsignedChar()
    checker_source.SetNumberOfScalarComponents(3)
    checker_source.SetExtent(0, 255, 0, 255, 0, 0)
    tile_size = 32
    for row in range(8):
        for column in range(8):
            color = 235 if (row + column) % 2 == 0 else 45
            checker_source.SetDrawColor(color, color, color)
            checker_source.FillBox(
                column * tile_size,
                (column + 1) * tile_size - 1,
                row * tile_size,
                (row + 1) * tile_size - 1,
            )
    checker_source.Update()

    checker_texture = vtk.vtkTexture()
    checker_texture.SetInputConnection(checker_source.GetOutputPort())
    checker_texture.InterpolateOff()
    checker_texture.RepeatOn()

    model_mapper = vtk.vtkPolyDataMapper()
    model_mapper.SetInputData(result_mesh)
    model_mapper.ScalarVisibilityOff()
    model_actor = vtk.vtkActor()
    model_actor.SetMapper(model_mapper)
    model_actor.SetTexture(checker_texture)
    model_actor.GetProperty().EdgeVisibilityOn()
    model_actor.GetProperty().SetEdgeColor(0.15, 0.15, 0.15)
    model_actor.GetProperty().SetLineWidth(0.5)

    uv_mesh = vtk.vtkPolyData()
    uv_mesh.DeepCopy(result_mesh)
    uv_points = vtk.vtkPoints()
    uv_points.SetNumberOfPoints(result_mesh.GetNumberOfPoints())
    for point_index in range(result_mesh.GetNumberOfPoints()):
        u, v = texture_coordinates.GetTuple2(point_index)
        uv_points.SetPoint(point_index, u, v, 0.0)
    uv_mesh.SetPoints(uv_points)

    uv_mapper = vtk.vtkPolyDataMapper()
    uv_mapper.SetInputData(uv_mesh)
    uv_mapper.ScalarVisibilityOff()
    uv_actor = vtk.vtkActor()
    uv_actor.SetMapper(uv_mapper)
    uv_actor.GetProperty().SetColor(0.72, 0.84, 0.95)
    uv_actor.GetProperty().EdgeVisibilityOn()
    uv_actor.GetProperty().SetEdgeColor(0.08, 0.08, 0.08)
    uv_actor.GetProperty().SetLineWidth(0.7)

    model_renderer = vtk.vtkRenderer()
    model_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)
    model_renderer.SetBackground(0.12, 0.15, 0.19)
    model_renderer.AddActor(model_actor)
    model_renderer.AddActor(cut_actor)
    model_renderer.ResetCamera()

    uv_renderer = vtk.vtkRenderer()
    uv_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
    uv_renderer.SetBackground(0.96, 0.96, 0.96)
    uv_renderer.AddActor(uv_actor)
    uv_renderer.AddActor(uv_cut_actor)
    uv_renderer.ResetCamera()

    def add_title(renderer, title: str, color: tuple[float, float, float]) -> None:
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(title)
        text_actor.GetTextProperty().SetFontSize(22)
        text_actor.GetTextProperty().SetColor(*color)
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.SetPosition(0.03, 0.94)
        renderer.AddActor2D(text_actor)

    add_title(model_renderer, "OptCuts 3D result", (1.0, 1.0, 1.0))
    add_title(uv_renderer, "OptCuts UV mesh", (0.08, 0.08, 0.08))

    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Official OptCuts Result - VTK")
    render_window.SetSize(1400, 760)
    render_window.AddRenderer(model_renderer)
    render_window.AddRenderer(uv_renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    render_window.Render()
    interactor.Initialize()
    interactor.Start()


def main() -> None:
    stl_path = select_stl_file()
    if stl_path is None:
        return

    print("Converting STL and restoring its shared triangle topology...", flush=True)
    input_obj = convert_stl_to_obj(stl_path)
    result_obj = run_official_optcuts(input_obj)
    show_result_with_vtk(result_obj)


if __name__ == "__main__":
    main()
#基础
