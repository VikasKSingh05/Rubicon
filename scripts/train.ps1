# Rubicon training entrypoint for PowerShell
$Root = Split-Path -Parent $PSScriptRoot
Push-Location "$Root\ai-engine"
try {
    python train.py @args
} finally {
    Pop-Location
}
