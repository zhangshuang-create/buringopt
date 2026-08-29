@echo off
setlocal
set "PYTHON_EXE=E:\AI-DL\Anoconda2020\envs\XYJpytorch\python.exe"
set "SCRIPT_DIR=%~dp0"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_DIR%Geodesics.py" --render-backend pyvista --subdivide 1 --mds-max-points 4000 --dense-neighbors 16 --uv-average-points-per-face 12 %*
endlocal
