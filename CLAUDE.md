# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NTHU 114學年第2學期 LLM Security System 課程 Week 6 作業：Advanced RAG 系統實作。使用 LangChain + FAISS + HuggingFace 建構檢索增強生成系統，支援兩組知識庫（HF官方文件 / 資安資料集）與兩種 LLM 模式（本地 Ollama / 雲端 LiteLLM）。

## Commands

```bash
uv sync                  # 安裝所有依賴至 .venv
ollama pull gpt-oss:20b  # 下載本地 LLM 模型（需先安裝 Ollama）
```

## Architecture

主要實作在 `main.ipynb`，依照 `docs/RAG_SDD.md` 的模組設計：

- **Cell 1 集中設定**：`USE_LOCAL_DATA` / `USE_LOCAL_LLM` 旗標控制知識庫與 LLM 切換
- **Cell 2-4 離線索引**：載入知識庫 → RecursiveCharacterTextSplitter 切分（token-based, 512） → gte-small 嵌入 → FAISS 向量庫
- **Cell 5-7 組件載入**：LiteLLM 統一 LLM 介面 / Prompt 模板（中英雙語）/ CrossEncoder 重排序
- **Cell 8-9 線上查詢**：`answer_with_rag()` 串接檢索→重排→生成

## Key Design Decisions

- LLM 統一透過 LiteLLM 呼叫，本地 Ollama 用 `ollama/gpt-oss:20b`，雲端用 `gemini/gemini-2.5-flash`
- 嵌入模型 `thenlper/gte-small`（384維），切分以 token 數計算而非字元數
- 資安資料集載入：MD 檔整份讀入後切分，CSV 逐筆拼接欄位為 LangchainDocument

## Data Files (data/)

- `knowledge_base.md` — Prompt Injection 知識庫（8種攻擊，30筆）
- `phishing_knowledge_base.md` — 釣魚郵件知識庫
- `prompt_injection_dataset.csv` — 結構化 PI 資料（30筆）
- `phishing_email_dataset.csv` / `.json` — 釣魚郵件資料（15筆）

## Documentation

- `docs/RAG_SDD.md` — SDD 設計文件（Notebook 實作藍圖）
- `docs/Advanced_RAG_Summary.md` — RAG 理論參考文件

## Language

文件與註解使用繁體中文。
