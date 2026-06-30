#!/usr/bin/env pwsh
# stop_svo_9am.ps1 — 排程於 09:00 停止 SVO 抽取並記錄進度

$BASE     = "C:\Users\666\Desktop\智慧知識庫"
$PID_FILE = "$BASE\svo_pid.txt"
$PROG_FILE= "$BASE\chunk_store\3de0f63b-b7b3-46ed-8c52-603766752fd0\f91d31e7-13f3-4f67-bd3d-a808a47fbd7e\svo_progress.json"
$MEM_FILE = "C:\Users\666\.claude\projects\C--Users-666-Desktop------\memory\project_kg_build_progress.md"
$RECORD   = "$BASE\svo_stop_record.md"
$TOTAL    = 2416

# ── 1. 停止 Python 程序 ──
$savedPid = if (Test-Path $PID_FILE) { [int](Get-Content $PID_FILE -Raw).Trim() } else { 0 }
if ($savedPid -and (Get-Process -Id $savedPid -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
    Write-Host "Killed PID $savedPid"
}
# 保險：清除所有殘留 build_kg python
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*run_build_kg*" } |
    Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 3

# ── 2. 讀取進度 ──
$done      = 0
$maxIdx    = 0
if (Test-Path $PROG_FILE) {
    $data  = Get-Content $PROG_FILE -Raw | ConvertFrom-Json
    $done  = ($data.PSObject.Properties | Where-Object { $_.Value.processed -eq $true }).Count
    $keys  = $data.PSObject.Properties.Name | ForEach-Object { [int]$_ } | Sort-Object
    $maxIdx= if ($keys.Count -gt 0) { $keys[-1] } else { 0 }
}
$remaining = $TOTAL - $done
$pct       = [Math]::Round($done / $TOTAL * 100, 1)

# ── 3. 寫入停止記錄 ──
$now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$record = @"
# SVO 停止記錄

**停止時間**：$now
**KG**：AI影音內容創作（3de0f63b-b7b3-46ed-8c52-603766752fd0）
**文件**：遊戲化實戰全書（f91d31e7-13f3-4f67-bd3d-a808a47fbd7e）

| 項目 | 數值 |
|------|------|
| 總 chunk 數 | $TOTAL |
| 已完成 | $done |
| 剩餘 | $remaining |
| 完成率 | $pct% |
| 最大已處理 index | $maxIdx |

## 繼續執行

```powershell
cd "C:\Users\666\Desktop\智慧知識庫"
python run_build_kg.py --kg 3de0f63b-b7b3-46ed-8c52-603766752fd0
```
"@
$record | Set-Content $RECORD -Encoding UTF8
Write-Host "Stop record written: $RECORD"

# ── 4. 更新 memory ──
if (Test-Path $MEM_FILE) {
    $content = Get-Content $MEM_FILE -Raw
    $content = $content -replace '(?m)^\*\*已完成 chunk\*\*：約 \*\*\d+\*\*.*$',
        "**已完成 chunk**：$done（截至 $now）"
    $content = $content -replace '(?m)^\*\*剩餘\*\*：約 \*\*[\d,]+\*\* 段$',
        "**剩餘**：約 **$remaining 段**"
    $content | Set-Content $MEM_FILE -Encoding UTF8
    Write-Host "Memory updated: $MEM_FILE"
}

Write-Host "=== SVO stop complete at $now: done=$done / $TOTAL ==="
