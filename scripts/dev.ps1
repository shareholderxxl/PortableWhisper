# Development script for Whisper4Windows
# Runs both frontend (Tauri) and backend (Python) in development mode

Write-Host "🚀 Starting Whisper4Windows Development Environment" -ForegroundColor Green

# Check if Tauri CLI is available
if (-not (Get-Command "cargo-tauri" -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Tauri CLI not found. Installing..." -ForegroundColor Yellow
    cargo install tauri-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install Tauri CLI" -ForegroundColor Red
        exit 1
    }
}

# Start backend server (once we have it)
Write-Host "📡 Starting backend server..." -ForegroundColor Blue
# cd backend
# python -m uvicorn main:app --reload --host 127.0.0.1 --port 8765 &

# Start frontend in dev mode
Write-Host "🖥️ Starting frontend..." -ForegroundColor Blue
cd frontend
cargo tauri dev

Write-Host "✅ Development environment ready!" -ForegroundColor Green