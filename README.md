# NIME PDF Artifact and Site Extractor (`extractor.py`) — README

This script batch-processes NIME paper PDFs from a folder, extracts whether each paper presents an NIME **artifact** (instrument/interface/controller/system/software/etc.), and writes both a structured **JSON** file and a human-readable **TXT** summary per PDF.

It uses **Ollama running locally (as a localhost service)** to call a **cloud DeepSeek model** that you’ve selected/configured inside Ollama. The extraction is done via a simple map-reduce flow (chunk-level extraction + final merge/dedup).

---

## 1) High-level workflow

1. Install and start **Ollama** on your machine (it provides a local service so that you don't need to rely on api or keys).
2. In your local Ollama:
   - **Sign in**,
   - Select or enable the **cloud DeepSeek** model you want to use (e.g. here we used `deepseek-v3.1:671b-cloud`).
3. Run `extractor.py`:
   - The Python script will connect to **local Ollama** via `ollama.Client()` as the localhost,
   - Local Ollama then uses your configured **cloud DeepSeek** model to answer the prompts,
   - The script extracts PDF text, chunks it, runs per-chunk extraction (map), then merges results (reduce),
   - Outputs `.json` + `.txt` files to the output folder.

---

## 2) Requirements

### Python
- Recommended: Python 3.10+ (3.11/3.12 typically OK)

### Ollama
- Ollama installed and running locally
- You must be able to use your chosen cloud DeepSeek model from Ollama

### Python packages
Third-party libraries used by the script:
- **PyMuPDF** (`fitz`) — PDF text extraction
- **pydantic** — schema validation / structured outputs
- **ollama** — Python client for talking to local Ollama

To use this script, please use this code to install libraries:
pip install PyMuPDF pydantic ollama

