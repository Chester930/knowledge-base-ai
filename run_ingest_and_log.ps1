#!/usr/bin/env pwsh
# run_ingest_and_log.ps1 — 自動跑 Ingestion 並在終止時記錄結果

$KG_ID      = "839aa61d-8d97-4e2a-8c74-10fa111c3f38"
$STAGING    = ".\workspace\claude_memory_staging"
$LOG_FILE   = ".\ingest_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$RESULT_MD  = ".\ingest_results.md"

function Write-Log {
    param([string]$msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content $LOG_FILE $line -Encoding UTF8
}

Write-Log "=== Ingestion 開始 ==="
Write-Log "KG ID: $KG_ID"
Write-Log "Staging: $STAGING"
Write-Log "檔案數: $((Get-ChildItem $STAGING -Filter '*.md').Count)"

$startTime = Get-Date

# 執行 ingestion
try {
    python run_ingest.py $STAGING --kg $KG_ID 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
    $exitCode = $LASTEXITCODE
} catch {
    Write-Log "例外: $_"
    $exitCode = 1
}

$endTime  = Get-Date
$elapsed  = ($endTime - $startTime).TotalMinutes

Write-Log "=== Ingestion 結束 ==="
Write-Log "退出碼: $exitCode"
Write-Log "耗時: $([Math]::Round($elapsed, 1)) 分鐘"

# 統計結果
$successCount = (Select-String -Path $LOG_FILE -Pattern "SUCCESS|成功|✓" -ErrorAction SilentlyContinue).Count
$failCount    = (Select-String -Path $LOG_FILE -Pattern "FAIL|失敗|ERROR" -ErrorAction SilentlyContinue).Count

Write-Log "成功: $successCount | 失敗: $failCount"

# 驗證 Neo4j
Write-Log "驗證 Neo4j Document 節點..."
try {
    $cred     = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("neo4j:kg_test_2024"))
    $body     = '{"statements":[{"statement":"MATCH (d:Document) RETURN count(d) as cnt"},{"statement":"MATCH (n:ConceptNode) RETURN count(n) as cnt2"}]}'
    $response = Invoke-RestMethod -Method Post `
        -Uri "http://localhost:7475/db/neo4j/tx/commit" `
        -ContentType "application/json" `
        -Headers @{Authorization = "Basic $cred"} `
        -Body $body
    $docCount     = $response.results[0].data[0].row[0]
    $conceptCount = $response.results[1].data[0].row[0]
    Write-Log "Neo4j Document: $docCount | ConceptNode: $conceptCount"
} catch {
    Write-Log "Neo4j 查詢失敗: $_"
    $docCount     = "N/A"
    $conceptCount = "N/A"
}

# 快速 RAG 驗收
Write-Log "執行 RAG 驗收查詢..."
try {
    $ragBody = '{"question":"claude-desktop Teams 系統的記憶共享機制","top_k":3}'
    $ragResp = Invoke-RestMethod -Method Post `
        -Uri "http://127.0.0.1:8000/agent/query" `
        -ContentType "application/json; charset=utf-8" `
        -Body ([Text.Encoding]::UTF8.GetBytes($ragBody)) `
        -TimeoutSec 60
    $ragStatus  = if ($ragResp.context.Count -gt 0) { "✅ 有召回 ($($ragResp.context.Count) 筆)" } else { "⚠️ 無召回（記憶未進 KG 或 index 未建）" }
    $ragSources = ($ragResp.sources -join ", ")
    Write-Log "RAG 狀態: $ragStatus"
    Write-Log "RAG 來源: $ragSources"
} catch {
    $ragStatus  = "❌ 查詢失敗: $_"
    $ragSources = ""
    Write-Log $ragStatus
}

# 寫入 Markdown 結果摘要
$summary = @"
# Ingestion 執行結果

**執行時間**：$($startTime.ToString('yyyy-MM-dd HH:mm:ss')) → $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))
**耗時**：$([Math]::Round($elapsed, 1)) 分鐘
**退出碼**：$exitCode
**KG ID**：$KG_ID

## 結果

| 項目 | 數值 |
|------|------|
| 成功 | $successCount |
| 失敗 | $failCount |
| Neo4j Document | $docCount |
| Neo4j ConceptNode | $conceptCount |
| RAG 驗收 | $ragStatus |
| RAG 來源 | $ragSources |

## 下一步

- [ ] 確認 Document 節點數正確（應為 39）
- [ ] 若 RAG 無召回，檢查 ConceptNode 向量索引
- [ ] 執行 ``run_build_kg.py`` 更新 SVO 圖譜

---
*Log 檔：$LOG_FILE*
"@

Set-Content $RESULT_MD $summary -Encoding UTF8
Write-Log "結果摘要已寫入：$RESULT_MD"
Write-Log "=== 完成 ==="
