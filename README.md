# Advanced RAG 系統與合成資料生成

NTHU 114 學年第 2 學期 — LLM Security System  
Week 6 作業 — TASK 2 & TASK 3

> 本作品為與 AI 共同協作的成果。

---

## 架構概覽

### Task 2：Advanced RAG 系統（`main.ipynb`）

採用兩階段檢索的 Advanced RAG 架構：

```
知識庫載入 → Token-based 切分 → 嵌入向量化（gte-small）→ FAISS 索引
                                                              ↓
使用者問題 → FAISS 粗檢索（top-30）→ CrossEncoder 精排（top-5）→ Prompt 組裝 → LLM 生成答案
```

### Task 3：Synthetic Data 生成（`task3_synthetic_data.ipynb`）

使用 LLM 自動生成合成釣魚信件資料集：

```
生成 Prompt → LLM（Ollama gpt-oss:20b）→ JSON 解析 → Schema 驗證 → 儲存 JSON/CSV
```

---

## 技術堆疊

| 組件 | 技術選擇 | 說明 |
|------|---------|------|
| 框架 | LangChain | 文檔載入、切分、向量庫整合 |
| 嵌入模型 | `thenlper/gte-small`（384 維） | L2 歸一化，點積 = 餘弦相似度 |
| 向量資料庫 | FAISS | 餘弦相似度檢索 |
| 重排序 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder 聯合編碼 |
| LLM | Ollama `gpt-oss:20b` / LiteLLM 雲端 | 統一介面，一鍵切換 |
| 視覺化 | PaCMAP + Plotly | 嵌入降維互動散點圖 |

---

## 知識庫

支援兩組知識庫（透過 `USE_LOCAL_DATA` 旗標切換）：

- **資安資料集**（`data/`）：Prompt Injection 知識庫（8 種攻擊、30 筆）+ 釣魚信件知識庫（15 筆）
- **HuggingFace 官方文件**：`m-ric/huggingface_doc`（備用）

---

## 快速開始

### 前置條件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 套件管理器
- [Ollama](https://ollama.com/download)（本地 LLM 模式）

### 安裝與執行

```bash
# 1. 安裝依賴
uv sync

# 2. 下載本地 LLM 模型
ollama pull gpt-oss:20b

# 3. 執行 Task 2：Advanced RAG
jupyter notebook main.ipynb

# 4. 執行 Task 3：合成資料生成
jupyter notebook task3_synthetic_data.ipynb
```

---

## Notebook 結構

### `main.ipynb` — Advanced RAG 系統（Task 2）

依序執行 Cell 0 ~ Cell 9 即可完成完整 RAG 流程。

| Cell | 功能 | 說明 |
|------|------|------|
| Cell 0 | 環境驗證 | 檢查 13 個必要套件 |
| Cell 1 | 集中設定 | `USE_LOCAL_DATA` / `USE_LOCAL_LLM` 旗標 |
| Cell 2 | 載入知識庫 | Markdown + CSV → `LangchainDocument`，共 47 篇 |
| Cell 3 | 文檔切分 | Token-based 切分，`chunk_size=512`，64 個片段 |
| Cell 4 | 建立向量庫 | gte-small 嵌入 → FAISS 索引 |
| Cell 4.5 | 嵌入視覺化 | PaCMAP 降維 + Plotly 散點圖 |
| Cell 5 | LLM 閱讀器 | LiteLLM 統一呼叫（本地 / 雲端） |
| Cell 6 | Prompt 模板 | 中英雙語，自動選擇 |
| Cell 7 | 重排序模型 | CrossEncoder 精排 |
| Cell 8 | RAG Pipeline | `answer_with_rag()` 完整流水線 |
| Cell 9 | 執行查詢 | 資料集專屬問題驗證 RAG 效果 |

### `task3_synthetic_data.ipynb` — 合成釣魚資料生成（Task 3）

依序執行 Cell 0 ~ Cell 5 即可生成 15 筆合成釣魚信件並儲存。

| Cell | 功能 | 說明 |
|------|------|------|
| Cell 0 | 環境設定 | LiteLLM + Ollama 設定（`temperature=0.8`、`max_tokens=8192`） |
| Cell 1 | 定義 Prompt | 指定 schema 與生成要求（多語言、多攻擊類型） |
| Cell 2 | 呼叫 LLM | `litellm.completion()` 生成 15 筆合成資料 |
| Cell 3 | 解析與驗證 | JSON 解析 + 10 欄位型別檢查 + email RFC 5321 格式驗證 |
| Cell 4 | 儲存檔案 | 輸出 `synthetic_phishing_dataset.json` + `.csv`，附統計摘要 |
| Cell 5 | 品質分析 | 攻擊類型 / 語言多樣性檢查 + 逐筆預覽 |

---

## 專案結構

```
.
├── main.ipynb                          # Task 2：Advanced RAG 系統
├── task3_synthetic_data.ipynb          # Task 3：合成釣魚資料生成
├── pyproject.toml                      # 專案依賴定義
├── data/
│   ├── knowledge_base.md               # Prompt Injection 知識庫
│   ├── phishing_knowledge_base.md      # 釣魚信件知識庫
│   ├── prompt_injection_dataset.csv    # PI 結構化資料（30 筆）
│   ├── phishing_email_dataset.csv      # 釣魚信件資料（15 筆）
│   ├── phishing_email_dataset.json     # 釣魚信件資料（JSON 格式）
│   ├── synthetic_phishing_dataset.json # Task 3 生成的合成資料（15 筆）
│   └── synthetic_phishing_dataset.csv  # Task 3 生成的合成資料（CSV）
└── docs/
    ├── RAG_SDD.md                      # SDD 設計文件
    └── Advanced_RAG_Summary.md         # RAG 理論參考
```
