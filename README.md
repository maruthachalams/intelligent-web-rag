# 🤖 Intelligent Web-RAG Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![GitHub license](https://img.shields.io/github/license/maruthachalams/intelligent-web-rag.svg)](https://github.com/maruthachalams/intelligent-web-rag/blob/main/LICENSE)

> A real-time, recursive multi-page Web Crawler and Retrieval-Augmented Generation (RAG) agent that goes beyond searching—it understands. Ingest any landing page or live-updating feed, and the pipeline automatically crawls nested article pages, splits and vectorizes the content into local persistent memory, and answers questions using grounded, citations-linked inference.

This project represents a complete **upgrade from standard web scraping to production-grade MLOps**—bridging the gap between raw data collection and secure, low-latency, contextual AI inference.

---

## 🌟 Features

* **🕸️ Asynchronous Depth-1 Recursive Crawler:** Combines `Requests` and `Headless Selenium` (with automated fallback if blocked) to scrape indices and recursively ingest nested child pages on the fly (perfect for live feeds like Yahoo Finance/News portals).
* **🧠 Windows-Safe Local Memory (ChromaDB):** Generates semantic vector indices using HuggingFace's `all-MiniLM-L6-v2` embeddings, storing data in local persistent storage. Includes a custom DB-level reset sequence to bypass OS file-locks on Windows (`WinError 32`).
* **🛰️ Async Gateway (FastAPI):** A high-performance async API orchestrating the ingest, crawl, and query pipelines. Fully integrates with Open-Weight LLMs (using Groq's high-speed developer endpoints like `openai/gpt-oss-20b`).
* **🎈 Visual Ingest Boundaries (Streamlit):** An interactive UI featuring live chunk-loading diagnostics, balloons, and boundary feedback listing exactly which nested links were safely digested and search-active.

---

## 📁 Repository Structure

```text
intelligent-web-rag/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI Gateway (orchestration & api endpoints)
│   ├── rag_engine.py    # Embeddings, chunking, and Windows-safe ChromaDB routines
│   └── scraper.py       # Recursive, same-domain child link and fallback scrapers
├── frontend/
│   └── app.py           # Streamlit user interface & diagnostic boundaries
├── .gitignore           # Prevents committing massive /venv, /chroma_db, and secret env files
├── .env                 # API Credentials (local setup)
├── requirements.txt     # Python dependencies
└── README.md            # You are here!
