@echo off
REM ============================================================
REM  run_slice_export.bat  -  GAN-PH slice exporter (for Juan)
REM
REM  USAGE: drag a .vtk file (or a folder containing .vtk files)
REM  onto this .bat file. Slices are written to a new folder
REM  named "<name>_slices" next to the input.
REM
REM  If ParaView is installed somewhere else, edit PVPYTHON below.
REM ============================================================

set "PVPYTHON=C:\Program Files\ParaView 6.1.1\bin\pvpython.exe"

if not exist "%PVPYTHON%" set "PVPYTHON=C:\ParaView\bin\pvpython.exe"
if not exist "%PVPYTHON%" (
    echo Could not find pvpython.exe.
    echo Edit this .bat file in Notepad and set PVPYTHON to your
    echo ParaView install, e.g. C:\...\ParaView 6.1.1\bin\pvpython.exe
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drag a .vtk file ^(or a folder of .vtk files^) onto this .bat file.
    pause
    exit /b 1
)

echo Using: %PVPYTHON%
echo Input: %~1
echo Output: %~dpn1_slices
echo.

"%PVPYTHON%" "%~dp0paraview_slice_export.py" "%~1" -o "%~dpn1_slices"

echo.
if errorlevel 1 (
    echo Something went wrong - send a screenshot of the messages above.
) else (
    echo Done! Slices are in: %~dpn1_slices
)
pause
