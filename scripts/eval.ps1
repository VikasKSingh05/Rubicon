# Rubicon evaluation entrypoint for PowerShell
$Root = Split-Path -Parent $PSScriptRoot
Push-Location "$Root\ai-engine"
try {
    python eval.py @args
} finally {
    Pop-Location
}
