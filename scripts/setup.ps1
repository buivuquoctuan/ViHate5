param(
    [string]$PythonPath = "python",
    [switch]$CpuOnly
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $PythonPath -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Cannot create venv. Pass -PythonPath with a working Python 3.12 executable." }
}
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TorchIndex = if ($CpuOnly) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu121" }
& $ProjectPython -m pip install --cache-dir .cache/pip "torch==2.5.1" --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $ProjectPython -m pip install --cache-dir .cache/pip -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }
& $ProjectPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $ProjectPython -m pip freeze --exclude-editable | Out-File -LiteralPath requirements-lock.txt -Encoding utf8
