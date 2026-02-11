import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from ollama import Client


# -----------------------------
# Configuration
# -----------------------------
MODEL = "deepseek-v3.1:671b-cloud"  # your Ollama model name
INPUT_DIR = Path(r"XXXXXXXXXXXXX")          # path: folder of NIME PDFs
OUTPUT_DIR = Path(r"XXXXXXXXXXXXX")          # path: output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunking by characters (simple + robust). Tune if needed.
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 1200

# Safety: limit quote sizes you store in outputs
MAX_QUOTE_CHARS = 400


# -----------------------------
# Schemas for structured output
# -----------------------------
class Artifact(BaseModel):
    name: Optional[str] = None
    genre: Optional[str] = Field(
        default=None,
        description="Artifact genre/type: e.g., instrument, controller, interface, installation, system, software, toolkit, wearable, etc."
    )
    evidence_quote: Optional[str] = None
    evidence_page: Optional[int] = None
    deployed_site: Optional[str] = Field(
        default=None,
        description="Where it is deployed/used: e.g., concert performance, gallery, public space, classroom, lab study, online, home, rehearsal, etc."
    )
    deployed_site_quote: Optional[str] = None
    deployed_site_page: Optional[int] = None


class PaperExtraction(BaseModel):
    has_artifact: bool
    artifacts: List[Artifact] = Field(default_factory=list)
    notes: Optional[str] = None


# -----------------------------
# PDF text extraction utilities
# -----------------------------
def extract_pages_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1_based, text).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # Normalize whitespace a bit
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append((i + 1, text))
    doc.close()
    return pages


def pages_to_marked_text(pages: List[Tuple[int, str]]) -> str:
    """
    Concatenate pages with explicit page markers so the model can cite pages.
    """
    parts = []
    for pno, txt in pages:
        if not txt:
            continue
        parts.append(f"\n\n=== PAGE {pno} ===\n{txt}")
    return "".join(parts).strip()


def chunk_text(marked_text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sliding-window chunking over characters.
    Keeps page markers inside chunks.
    """
    if len(marked_text) <= chunk_size:
        return [marked_text]

    chunks = []
    start = 0
    while start < len(marked_text):
        end = min(len(marked_text), start + chunk_size)
        chunk = marked_text[start:end]

        # Try not to cut mid-page-marker if possible
        # (best-effort: extend end to include a marker line break boundary)
        chunks.append(chunk)

        if end == len(marked_text):
            break
        start = max(0, end - overlap)

    return chunks


def clamp_quote(s: Optional[str], max_chars: int = MAX_QUOTE_CHARS) -> Optional[str]:
    if not s:
        return s
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def parse_page_from_quote_context(quote: Optional[str]) -> Optional[int]:
    """
    Not used directly; page is best extracted by instructing the model to cite it.
    Placeholder in case you later want heuristic page inference.
    """
    return None


# -----------------------------
# Ollama chat helpers
# -----------------------------
def chat_json(client: Client, model: str, messages: List[Dict[str, str]], stream: bool = False) -> str:
    """
    Returns full assistant content as a string.
    If stream=True, prints tokens while collecting.
    """
    if not stream:
        resp = client.chat(model, messages=messages)
        return resp["message"]["content"]

    buf = []
    for part in client.chat(model, messages=messages, stream=True):
        token = part["message"]["content"]
        print(token, end="", flush=True)
        buf.append(token)
    print()
    return "".join(buf)


JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Tries to extract JSON from a ```json ...``` block; falls back to first {...}.
    """
    m = JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))

    # Fallback: find the first JSON-like object
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(text[first:last + 1])

    raise ValueError("Could not locate a JSON object in model output.")


# -----------------------------
# Prompts
# -----------------------------
SYSTEM_PROMPT = """You are a careful research assistant extracting information from NIME papers.
You must only use evidence from the provided text. If uncertain, say uncertain.
Return STRICT JSON only, wrapped in a ```json``` code block, matching the requested schema.
Keep quotes short and include page numbers when asked.
"""

MAP_PROMPT = """Task:
From the excerpt below, extract whether the paper presents an NIME artifact and, if so, details.

Definitions:
- "Artifact" includes: instrument/interface/controller/device, installation, system, software/toolkit, wearable, methodology prototype, or a composed interactive setup if presented as a contribution.
- "Deployed site" means where it is used/shown/studied (e.g., performance/concert, gallery, public space, classroom, lab study, online, home).

Rules:
- If there is no evidence of an artifact, set has_artifact=false and artifacts=[].
- If there is evidence, set has_artifact=true and list each distinct artifact mentioned in this excerpt.
- For each artifact, include:
  - name (if present)
  - genre/type
  - evidence_quote and evidence_page
  - deployed_site (if present in excerpt)
  - deployed_site_quote and deployed_site_page
- If deployed site is not stated in the excerpt, leave deployed_site fields null.

Return schema:
{
  "has_artifact": boolean,
  "artifacts": [
    {
      "name": string|null,
      "genre": string|null,
      "evidence_quote": string|null,
      "evidence_page": integer|null,
      "deployed_site": string|null,
      "deployed_site_quote": string|null,
      "deployed_site_page": integer|null
    }
  ],
  "notes": string|null
}

Excerpt:
"""

REDUCE_PROMPT = """You will be given partial extractions from multiple chunks of the same paper.
Your job:
- Merge/dedupe artifacts (same artifact may be repeated).
- Prefer entries with clearer evidence and page numbers.
- Decide final has_artifact.
- Output final JSON in the same schema.
- If there are conflicting sites, include the best-supported site, and mention ambiguity in notes.

Partial extractions JSON list:
"""


# -----------------------------
# Map-reduce extraction
# -----------------------------
def map_extract(client: Client, chunk: str) -> PaperExtraction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MAP_PROMPT + chunk}
    ]
    raw = chat_json(client, MODEL, messages, stream=False)
    obj = extract_json_object(raw)
    # validate
    pe = PaperExtraction(**obj)

    # clamp quotes
    for a in pe.artifacts:
        a.evidence_quote = clamp_quote(a.evidence_quote)
        a.deployed_site_quote = clamp_quote(a.deployed_site_quote)
    return pe


def reduce_merge(client: Client, partials: List[PaperExtraction]) -> PaperExtraction:
    partial_dicts = [p.model_dump() for p in partials]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REDUCE_PROMPT + json.dumps(partial_dicts, ensure_ascii=False)}
    ]
    raw = chat_json(client, MODEL, messages, stream=False)
    obj = extract_json_object(raw)
    pe = PaperExtraction(**obj)

    for a in pe.artifacts:
        a.evidence_quote = clamp_quote(a.evidence_quote)
        a.deployed_site_quote = clamp_quote(a.deployed_site_quote)
    return pe


# -----------------------------
# Output formatting
# -----------------------------
def format_txt(pdf_name: str, extraction: PaperExtraction) -> str:
    lines = []
    lines.append(f"Paper: {pdf_name}")
    lines.append(f"Has NIME artifact: {extraction.has_artifact}")
    lines.append("")

    if extraction.has_artifact and extraction.artifacts:
        lines.append("Artifacts:")
        for i, a in enumerate(extraction.artifacts, 1):
            lines.append(f"{i}. Name: {a.name or '(unspecified)'}")
            lines.append(f"   Genre: {a.genre or '(unspecified)'}")
            if a.evidence_quote:
                p = f"p.{a.evidence_page}" if a.evidence_page else "p.(unknown)"
                lines.append(f"   Evidence ({p}): {a.evidence_quote}")
            else:
                lines.append("   Evidence: (none captured)")

            if a.deployed_site:
                lines.append(f"   Deployed site: {a.deployed_site}")
                if a.deployed_site_quote:
                    p = f"p.{a.deployed_site_page}" if a.deployed_site_page else "p.(unknown)"
                    lines.append(f"   Site evidence ({p}): {a.deployed_site_quote}")
            else:
                lines.append("   Deployed site: (not stated)")
            lines.append("")
    else:
        lines.append("Artifacts: none found (based on extracted text).")
        lines.append("")

    if extraction.notes:
        lines.append("Notes:")
        lines.append(extraction.notes.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# -----------------------------
# Main
# -----------------------------
def process_pdf(client: Client, pdf_path: Path) -> PaperExtraction:
    pages = extract_pages_text(pdf_path)
    marked = pages_to_marked_text(pages)
    chunks = chunk_text(marked, CHUNK_SIZE, CHUNK_OVERLAP)

    partials: List[PaperExtraction] = []
    for idx, ch in enumerate(chunks, 1):
        try:
            pe = map_extract(client, ch)
            partials.append(pe)
        except Exception as e:
            # Keep going; you'll still get something from other chunks
            partials.append(PaperExtraction(has_artifact=False, artifacts=[], notes=f"Chunk {idx} failed: {e}"))

    final = reduce_merge(client, partials)
    return final


def main():
    client = Client()

    pdfs = sorted([p for p in INPUT_DIR.glob("*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {INPUT_DIR.resolve()}")

    for pdf_path in pdfs:
        print(f"\n=== Processing: {pdf_path.name} ===")
        extraction = process_pdf(client, pdf_path)

        # Write JSON
        json_path = OUTPUT_DIR / (pdf_path.stem + ".json")
        json_path.write_text(
            json.dumps(extraction.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Write TXT
        txt_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
        txt_path.write_text(format_txt(pdf_path.name, extraction), encoding="utf-8")

        print(f"Wrote: {txt_path.name}, {json_path.name}")

    print("\nDone.")


if __name__ == "__main__":

    main()
