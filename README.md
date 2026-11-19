# 🤖 Intelligent Web-RAG Agent

![CI Pipeline](https://github.com/maruthachalams/intelligent-web-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)

An end-to-end **Web Retrieval-Augmented Generation (RAG)** system engineered with production **MLOps / LLMOps** standards—featuring decoupled API serving, Docker containerization, automated LLM evaluation (Ragas), and GitHub Actions CI/CD.

---

## 🏗️ Architecture

```text
[ Streamlit UI ]  ──HTTP──▶  [ FastAPI + Uvicorn (Docker Container) ]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
        [ Web Scraper ]                           [ RAGEngine ]
    (Crawl & Chunk Live URLs)              (ChromaDB + all-MiniLM-L6-v2)
                  │                                         │
                  └──────────────▶ [ ChromaDB ] ◀───────────┘
                                       │
                                       ▼
                           [ Groq LLM Inference ]
                            (openai/gpt-oss-20b)
```

---

## 🚀 MLOps & LLMOps Highlights

* **Decoupled Serving & Containerization:** FastAPI + Uvicorn backend packaged in a lightweight `python:3.12-slim` Docker image, completely separated from the Streamlit frontend.
* **Automated LLM Evaluation (Ragas):** Evaluates retrieval and generation quality using an LLM-as-a-Judge (`openai/gpt-oss-120b`) instead of manual spot-checks.
* **CI/CD Quality Gates:** GitHub Actions workflow (`ci.yml`) automatically installs dependencies and runs the `pytest` suite on every push to `main`.
* **Live Web Ingestion:** Crawls and chunks live web pages on demand, indexing them into a persistent ChromaDB vector store.

---

## 📊 Automated Evaluation Results (Ragas)

Evaluated via `python -m backend.eval_ragas` using `openai/gpt-oss-120b` as the judge model:

| Metric | Score | What It Measures |
| :--- | :---: | :--- |
| **Faithfulness** | **0.6667** | Factual consistency of the generated answer against retrieved web context |
| **Answer Relevance** | **0.6967** | How directly and completely the answer addresses the user's query |

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend UI** | Streamlit |
| **Vector Store & Embeddings** | ChromaDB, `all-MiniLM-L6-v2` |
| **LLM Inference** | Groq (`openai/gpt-oss-20b`) |
| **LLM Evaluation** | Ragas (`openai/gpt-oss-120b` judge) |
| **Containerization** | Docker (`python:3.12-slim`) |
| **Testing & CI/CD** | Pytest, GitHub Actions |

---

## ⚡ Quick Start (Docker + Streamlit)

### 1. Clone & Configure Environment

```bash
git clone [https://github.com/maruthachalams/intelligent-web-rag.git](https://github.com/maruthachalams/intelligent-web-rag.git)
cd intelligent-web-rag
```

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Build & Run the FastAPI Backend in Docker

```bash
docker build -t intelligent-web-rag .
docker run -d --name rag-backend --env-file .env -p 8000:8000 intelligent-web-rag
```

Verify the API is live at `http://localhost:8000/docs`.

### 3. Launch the Streamlit Frontend

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

---

## 🧪 Running Tests & Evaluation Locally

**Run the Pytest suite:**

```bash
python -m pytest -v
```

**Run the Ragas evaluation pipeline:**

```bash
python -m backend.eval_ragas
```

---

## 🔧 Engineering Challenges Solved

* **Windows DLL Conflict (`onnxruntime` / `pyarrow`):** Resolved segmentation faults by controlling module import order and pinning `pyarrow<18` for NumPy 2.0+ compatibility.
* **CI/CD Path & Secret Handling:** Configured `python -m pytest` in GitHub Actions to resolve root module imports and integrated repository secrets for live inference testing.
* **Ragas Token-Usage Compatibility:** Built a lightweight wrapper around `langchain-groq` to handle nested token-usage aggregation cleanly during automated evaluation runs.
