# Rubicon serve script for PowerShell
param (
    [switch]$Local,
    [switch]$Build
)

$Root = Split-Path -Parent $PSScriptRoot

if ($Local) {
    Write-Host "Starting Rubicon services locally..." -ForegroundColor Cyan
    Write-Host "Starting AI Engine on port 8000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\ai-engine'; python -m uvicorn src.inference:app --host 0.0.0.0 --port 8000"
    
    Write-Host "Starting Backend on port 4000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\backend'; npm start"
    
    Write-Host "Starting Agent Service on port 8001..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\agent-service'; npm start"
    
    Write-Host "Starting Frontend on port 3000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; npm run dev"
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
