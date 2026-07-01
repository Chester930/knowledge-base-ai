# 智慧知識庫 — 一鍵啟動腳本
# 用法：右鍵 → 用 PowerShell 執行，或 pwsh start.ps1

Set-Location $PSScriptRoot

Write-Host "=== 智慧知識庫啟動中 ===" -ForegroundColor Cyan

# ── 1. 檢查 Docker Desktop ────────────────────────────────────────────────
Write-Host "`n[1/4] 檢查 Docker Desktop..." -ForegroundColor Yellow
try {
    docker info *>$null
    Write-Host "  Docker Desktop 正常" -ForegroundColor Green
} catch {
    Write-Host "  Docker Desktop 未啟動！請先開啟 Docker Desktop 再執行此腳本。" -ForegroundColor Red
    Read-Host "按 Enter 離開"
    exit 1
}

# ── 2. 檢查 Neo4j Desktop 是否仍在跑（會衝突）──────────────────────────
Write-Host "`n[2/4] 檢查 Neo4j Desktop DBMS..." -ForegroundColor Yellow
$neo4jPort = Test-NetConnection -ComputerName localhost -Port 7989 -WarningAction SilentlyContinue
if ($neo4jPort.TcpTestSucceeded) {
    Write-Host "  ⚠️  偵測到 port 7989 已被佔用（Neo4j Desktop 可能正在執行）" -ForegroundColor Red
    Write-Host "  請到 Neo4j Desktop → KG Test → Stop，再重新執行此腳本。" -ForegroundColor Red
    Read-Host "按 Enter 離開"
    exit 1
}
Write-Host "  Port 7989 空閒，可以啟動" -ForegroundColor Green

# ── 3. 檢查 Ollama ────────────────────────────────────────────────────────
Write-Host "`n[3/4] 檢查 Ollama（qwen2.5:7b）..." -ForegroundColor Yellow
try {
    $models = (Invoke-RestMethod "http://localhost:11434/api/tags").models.name
    if ($models -contains "qwen2.5:7b") {
        Write-Host "  Ollama + qwen2.5:7b 就緒" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Ollama 有在跑但找不到 qwen2.5:7b，嘗試拉取..." -ForegroundColor Yellow
        ollama pull qwen2.5:7b
    }
} catch {
    Write-Host "  ⚠️  Ollama 未回應，嘗試啟動..." -ForegroundColor Yellow
    Start-Process "ollama" "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# ── 4. 啟動 Docker Compose ────────────────────────────────────────────────
Write-Host "`n[4/4] 啟動 Neo4j + API..." -ForegroundColor Yellow
docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  啟動失敗，查看 log：docker compose logs" -ForegroundColor Red
    Read-Host "按 Enter 離開"
    exit 1
}

# ── 等待 API 就緒 ─────────────────────────────────────────────────────────
Write-Host "`n等待服務啟動（最多 90 秒）..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 18; $i++) {
    Start-Sleep -Seconds 5
    try {
        $health = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 3
        if ($health.status -eq "ok") { $ready = $true; break }
    } catch {}
    Write-Host "  等待中... $($($i+1)*5)s" -ForegroundColor Gray
}

if ($ready) {
    Write-Host "`n=== 啟動完成 ===" -ForegroundColor Green
    Write-Host "  前端：http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  Neo4j Browser：http://localhost:7475" -ForegroundColor Cyan
    Start-Process "http://localhost:8000"
} else {
    Write-Host "`n  API 尚未就緒，請執行 'docker compose logs api' 查看狀況" -ForegroundColor Yellow
}
