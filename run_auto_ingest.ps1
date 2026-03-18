param(
    [string]$PythonPath = "",
    [string]$WatchDir = "",
    [string]$Workspace = "",
    [int]$Stride = 90,
    [int]$MinFace = 48,
    [double]$Confidence = 0.45,
    [int]$ActivityGap = 20,
    [int]$MinActivity = 3,
    [switch]$Once,
    [switch]$Recursive,
    [switch]$ForcePolling
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $repoRoot "auto_ingest.py"

if ([string]::IsNullOrWhiteSpace($WatchDir)) {
    $WatchDir = Join-Path $repoRoot "drop_incoming"
}

if ([string]::IsNullOrWhiteSpace($Workspace)) {
    $Workspace = Join-Path $repoRoot "workspace"
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $localVenvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $localVenvPython) {
        $PythonPath = $localVenvPython
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw "Python not found. Supply -PythonPath or create .venv in the repo root."
        }
        $PythonPath = $pythonCommand.Source
    }
}

if (-not (Test-Path $WatchDir)) {
    New-Item -ItemType Directory -Path $WatchDir | Out-Null
}

if (-not (Test-Path $Workspace)) {
    New-Item -ItemType Directory -Path $Workspace | Out-Null
}

$commandArgs = @(
    $script,
    $WatchDir,
    "--workspace", $Workspace,
    "--stride", $Stride,
    "--min-face", $MinFace,
    "--confidence", $Confidence,
    "--activity-gap", $ActivityGap,
    "--min-activity", $MinActivity
)

if ($Once) {
    $commandArgs += "--once"
}

if ($Recursive) {
    $commandArgs += "--recursive"
}

if ($ForcePolling) {
    $commandArgs += "--force-polling"
}

& $PythonPath @commandArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

exit 0