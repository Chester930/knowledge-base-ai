# 路線圖 · Roadmap

> 最後更新：2026-06-21

本文件記錄「智慧知識庫」的長期願景與開發計畫。  
若你對某個階段的方向有興趣，歡迎填寫下方意願表單或直接開 Issue 討論。

---

## 願景

**任何人都能建立自己的知識圖譜，並選擇將知識公開給世界。**

個人維護的知識庫 → 標記公開 → 自動匯入世界知識圖譜 → 所有人皆可探索與提問。

---

## 現況（已完成）

### ✅ Phase 1 — 本機多場景知識圖譜

- **多 KG 管理**：建立多個場景知識圖譜，各自獨立
- **SVO 知識提取**：文件 → 30 種語意關係三元組 → Neo4j 圖譜
- **雙層 RAG 問答**：ConceptNode 路由層 + BFS 圖遍歷，精準問答
- **公開 / 私有 KG**：標記 `is_public`，World Agent 只讀公開 KG
- **World Agent**：跨所有公開 KG 問答，支援 SSE 串流
- **實體探索 UI**：搜尋實體 → 點擊展開鄰居關係 → 遞迴圖探索
- **多 Provider 支援**：Ollama / OpenAI / Anthropic / Gemini / Grok
- **語音 / 影片轉譯**：faster-whisper 支援 MP3 / MP4 / WAV / M4A 等格式
- **Docker 一鍵部署**、完整文件（SETUP / PROVIDERS / ARCHITECTURE / API）

---

## 計畫中

### 🔄 Phase 2 — 世界知識聯邦（World Knowledge Federation）

**核心問題**：Phase 1 的 World Agent 只能看到「本機上」的公開 KG。  
如何讓不同人、不同機器的公開知識互相流通？

#### 設計方向

採用 **「本地維護 + 雲端 Hub」混合架構**，結合兩種模式的優點：

| 模式 | 優點 | 缺點 |
|------|------|------|
| 純聯邦 API（各 instance 互查）| 去中心化，個人完全控制 | 對方離線就查不到 |
| 純雲端資料庫 | 永遠在線，高可用 | 需中心化管理、付費 |
| **混合（本計畫）** | 本地控制 + 雲端保證可用性 | 有同步延遲 |

#### 架構圖

```
你的 instance                  他的 instance
  ├── 私有 KG（不推送）
  └── 公開 KG ─────────────────────────────────┐
                                               ↓ 定期 push
                                         Cloud Neo4j Hub
                                         （聚合所有人的公開 KG）
                                               │
                                         World Agent API
                                         （永遠在線，所有人可查詢）
```

這就像 **Git + GitHub** 的關係：
- 本地 repo = 你的 instance，你完全控制
- GitHub = Cloud Hub，保存所有人推送的公開版本
- `git push` = 定期將公開 KG 同步到 Hub

#### 主要設計決策

**1. 同步格式**

公開 KG 匯出為 NDJSON（每行一條 SVO triple），定期推送到 Hub：

```json
{"subject": "Claude Code", "subject_type": "工具", "rel_type": "USES", "verb": "使用", "object": "Agent", "object_type": "概念", "kg_id": "xxx", "instance_id": "yyy"}
```

**2. 實體命名空間隔離**

不同 instance 的同名實體以 `instance_id` 區隔，避免知識混雜：

```
你的「Claude」(instance: alice) ≠ 他的「Claude」(instance: bob)
```

World Agent 可選擇「合併同名實體」或「保持各自獨立」。

**3. 同步頻率**

知識不需要實時更新。建議每日或每次手動標記公開時觸發推送。

#### 待解決問題

| 問題 | 說明 |
|------|------|
| Hub 由誰維護？ | 可由社群共同維護，或允許任何人自架 Hub |
| Cloud DB 費用 | Neo4j AuraDB 免費版 200K 節點；超過需評估成本分攤方式 |
| 實體同義詞合併 | `強化學習` 與 `Reinforcement Learning` 應視為同一實體嗎？ |
| 資料下架機制 | 如何從 Hub 撤回已推送的知識（隱私 / 勘誤）？ |

#### 實作計畫（草案）

```
Phase 2a：定義 push 協議
  → POST /world/sync  接收 NDJSON，寫入雲端 DB
  → 各 instance 可手動或定期推送公開 KG

Phase 2b：架設公共 Hub
  → 一台永遠在線的 instance 作為 Hub
  → 開放 GET /world/chat（所有人免登入可查詢公開知識）

Phase 2c：衝突解決與實體對齊
  → instance_id 命名空間（必做）
  → 可選：同義詞合併 Ontology alignment
```

---

### 💡 未來探索方向（尚未排期）

以下是社群提出或作者思考中的方向，尚未進入正式計畫：

- **KG 版本控制**：像 Git 一樣，追蹤知識的變更歷史
- **實體圖視覺化**：互動式力導向圖（D3.js / Cytoscape.js）
- **使用者帳號系統**：追蹤個人貢獻與知識授權
- **KG 訂閱 / 追蹤**：訂閱別人的公開 KG，自動同步更新
- **知識溯源**：每條 SVO 事實標記來源文件與信心分數
- **多語言實體對齊**：跨語言知識連結

---

## 參與意願調查

> **Google 表單連結（即將開放）**
>
> 如果你對 Phase 2「世界知識聯邦」有興趣，或想加入早期測試，  
> 請填寫意願調查表單。你的回應將直接影響開發優先順序。
>
> 📋 **[意願調查表單 — 連結待補]**

或直接在 GitHub 開 Issue 留言：[github.com/Chester930/knowledge-base-ai/issues](https://github.com/Chester930/knowledge-base-ai/issues)

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| v0.1 | 2026-06 | Phase 1 完成，公開 Phase 2 計畫草案 |
