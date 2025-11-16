@echo off
setlocal enabledelayedexpansion

echo PythonFileToYoutube dependency check (Windows CMD)
set missing=
set PYTHON_CMD=python

REM Check Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    set missing=!missing!\n- Python 3.11: Install Python 3.11 and ensure "python" is on PATH.
) else (
    for /f "tokens=2 delims= " %%a in ('%PYTHON_CMD% --version') do set PY_VERSION=%%a
    for /f "tokens=1,2 delims=." %%m in ("%PY_VERSION%") do (
        if %%m LSS 3 (set PY_OUTDATED=1) else if %%m EQU 3 if %%n LSS 11 set PY_OUTDATED=1
    )
    if defined PY_OUTDATED (
        set missing=!missing!\n- Python >= 3.11: Detected %PY_VERSION%. Upgrade recommended.
    ) else (
        echo ^> Python %PY_VERSION%
    )
)

REM PyTorch + CUDA check
%PYTHON_CMD% -c "import importlib.util^ 
state={'installed':False,'cuda':False,'version':None}^ 
spec=importlib.util.find_spec('torch')^ 
if spec is not None:^ 
    import torch^ 
    state['installed']=True^ 
    state['version']=getattr(torch,'__version__','unknown')^ 
    try:^ 
        state['cuda']=bool(torch.cuda.is_available())^ 
    except Exception:^ 
        state['cuda']=False^ 
print('INSTALLED=' + ('True' if state['installed'] else 'False'))^ 
print('CUDA=' + ('True' if state['cuda'] else 'False'))^ 
print('VERSION=' + (state['version'] or 'unknown'))" > "%TEMP%\torch_state.txt" 2>nul
if errorlevel 1 (
    set missing=!missing!\n- PyTorch: Install the CUDA build from pytorch.org.
) else (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%TEMP%\torch_state.txt") do (
        if /I "%%a"=="INSTALLED" set TORCH_INSTALLED=%%b
        if /I "%%a"=="CUDA" set TORCH_CUDA=%%b
        if /I "%%a"=="VERSION" set TORCH_VERSION=%%b
    )
    del "%TEMP%\torch_state.txt" >nul 2>&1
    if /I "!TORCH_INSTALLED!" NEQ "TRUE" (
        set missing=!missing!\n- PyTorch: Install the CUDA build from pytorch.org.
    ) else (
        if /I "!TORCH_CUDA!"=="TRUE" (
            echo ^> PyTorch !TORCH_VERSION! (CUDA available)
        ) else (
            echo ^> PyTorch !TORCH_VERSION! (CUDA NOT detected)
            set missing=!missing!\n- CUDA for PyTorch: torch.cuda.is_available() returned False. Install NVIDIA drivers/CUDA toolkit.
        )
    )
)

REM Helper to check external command
call :CheckCommand ffmpeg "FFmpeg" "Install FFmpeg and add it to PATH."
call :CheckCommand 7z "7-Zip CLI" "Install 7-Zip / p7zip and expose '7z'."
call :CheckCommand par2 "PAR2" "Install par2cmdline and expose 'par2'."
call :CheckCommand nvidia-smi "CUDA / NVIDIA driver" "Install NVIDIA drivers/CUDA toolkit so 'nvidia-smi' works."

if "%missing%"=="" (
    echo All dependencies satisfied.
    exit /b 0
)
echo.
echo Missing dependencies:
echo %missing%
exit /b 1

:CheckCommand
where %1 >nul 2>&1
if errorlevel 1 (
    set missing=!missing!\n- %2: %3
) else (
    echo ^> %2
)
exit /b 0
