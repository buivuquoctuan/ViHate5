param(
    [int]$Epochs = 50,
    [int]$BatchSize = 16,
    [int]$Seed = 42,
    [string]$RunName = ("baseline_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$Device = "cuda",
    [switch]$SkipDownload,
    [switch]$Robustness
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) { throw "Run scripts/setup.ps1 first." }
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw "RunName must contain only letters, digits, underscore and hyphen." }
$RunPath = Join-Path "runs" $RunName
foreach ($Artifact in @("best.pt", "history.csv", "config.json", "training_summary.json")) {
    if (Test-Path -LiteralPath (Join-Path $RunPath $Artifact)) {
        throw "Run already exists: $RunPath. Choose a new -RunName."
    }
}
function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $ProjectPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Project step failed: $($Arguments -join ' ')" }
}
if (-not $SkipDownload) {
    Invoke-ProjectPython -Arguments @("-m", "crnn.download")
}
if (-not (Test-Path -LiteralPath "data/manifests/split_metadata.json")) {
    Invoke-ProjectPython -Arguments @("-m", "crnn.prepare", "--seed", "42")
}
Invoke-ProjectPython -Arguments @("-m", "crnn.train", "--train", "data/manifests/train.csv", "--val", "data/manifests/val.csv", "--output", $RunPath, "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--seed", "$Seed", "--device", $Device)
$Checkpoint = Join-Path $RunPath "best.pt"
$CleanPath = Join-Path $RunPath "evaluation"
Invoke-ProjectPython -Arguments @("-m", "crnn.evaluate", "--checkpoint", $Checkpoint, "--manifest", "data/manifests/test.csv", "--output", $CleanPath, "--batch-size", "$BatchSize", "--device", $Device)
$PairPaths = @()
if ($Robustness) {
    foreach ($Kind in @("gaussian_blur", "perspective", "wave")) {
        foreach ($Level in @(1, 2, 3)) {
            $ConditionPath = Join-Path $RunPath "robustness/$($Kind)_$Level"
            Invoke-ProjectPython -Arguments @("-m", "crnn.evaluate", "--checkpoint", $Checkpoint, "--manifest", "data/manifests/test.csv", "--output", $ConditionPath, "--batch-size", "$BatchSize", "--device", $Device, "--corruption", $Kind, "--severity", "$Level")
            $PairPaths += Join-Path $ConditionPath "predictions.csv"
        }
    }
}
$AnalysisArguments = @("-m", "crnn.analyze", "--predictions", (Join-Path $CleanPath "predictions.csv"), "--output", (Join-Path $RunPath "analysis"), "--calibration", "data/manifests/train.csv", "--annotations", "data/manifests/test_annotations.csv")
if ($PairPaths.Count -gt 0) { $AnalysisArguments += "--paired-predictions"; $AnalysisArguments += $PairPaths }
Invoke-ProjectPython -Arguments $AnalysisArguments
Invoke-ProjectPython -Arguments @("-m", "crnn.error_patterns", "--predictions", (Join-Path $CleanPath "predictions.csv"), "--output", (Join-Path $RunPath "analysis"), "--train-manifest", "data/manifests/train.csv")
Invoke-ProjectPython -Arguments @("scripts/visualize_results.py", "--run", $RunPath)
Invoke-ProjectPython -Arguments @("scripts/write_results_summary.py", "--run", $RunPath, "--output", "docs/ket_qua_$RunName.md")
Write-Host "Completed. Report: $RunPath/analysis/report.md"
