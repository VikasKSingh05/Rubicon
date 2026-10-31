# Rubicon serve script for PowerShell
param (
    [switch]$Local,
    [switch]$Build
)

$Root = Split-Path -Parent $PSScriptRoot

# Ensure Docker is in PATH if Docker Desktop is installed in AppData or Program Files
$DockerPaths = @(
    "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
    "$env:ProgramFiles\Docker\Docker\resources\bin"
)
foreach ($dp in $DockerPaths) {
    if ((Test-Path $dp) -and ($env:Path -notlike "*$dp*")) {
        $env:Path = "$dp;$env:Path"
    }
}

if ($Local) {
    Write-Host "Starting Rubicon services locally..." -ForegroundColor Cyan
    Write-Host "Note: If AI engine Python dependencies are not installed locally, backend automatically uses deterministic AI stub." -ForegroundColor DarkGray
    
    Write-Host "Starting AI Engine on port 8000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd '$Root\ai-engine'; python -m uvicorn src.inference:app --host 0.0.0.0 --port 8000"
    
    Write-Host "Starting Backend on port 4000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd '$Root\backend'; cmd.exe /c npm start"
    
    Write-Host "Starting Agent Service on port 8001..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd '$Root\agent-service'; cmd.exe /c npm start"
    
    Write-Host "Starting Frontend on port 3000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", "cd '$Root\frontend'; cmd.exe /c npm run dev"
} else {
    Write-Host "Starting Rubicon stack with Docker Compose..." -ForegroundColor Cyan
    Push-Location $Root
    try {
        if ($Build) {
            docker compose up --build
        } else {
            docker compose up
        }
    } finally {
        Pop-Location
    }
}
