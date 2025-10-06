import hashlib
import json
import os
import pathlib
import re
from io import BytesIO
from typing import Any, Dict, Iterable, List

import chromadb
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --------------------------- Paths & Config ---------------------------

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"
INDEX_DIR = ROOT / "index"

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDER = SentenceTransformer(EMB_MODEL_NAME)

# --------------------------- Utilities ---------------------------

def checksum_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

HEAD_RE = re.compile(r"^\s*(Article|Art\.|Recital|Kapittel|Kap\.|Del|§)\b", flags=re.I)

def split_heading_aware(text: str) -> List[str]:
    """
    Split text whenever we encounter heading-like lines (Article, §, Kapittel, etc.).
    """
    chunks: List[str] = []
    cur: List[str] = []
    for ln in text.splitlines():
        if HEAD_RE.search(ln) and cur:
            chunks.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append("\n".join(cur).strip())
    return chunks

def semantic_chunk(block: str, target_tokens: int = 800, overlap_tokens: int = 100) -> List[str]:
    """
    Very lightweight chunker using word count as a proxy for tokens.
    Produces overlapping windows to preserve context.
    """
    words = block.split()
    if not words:
        return []
    tgt = max(100, target_tokens)
    ovl = max(0, min(overlap_tokens, tgt // 3))
    step = max(1, tgt - ovl)

    out: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + tgt)
        piece = " ".join(words[i:j]).strip()
        if piece:
            out.append(piece)
        if j >= n:
            break
        i += step
    return out

ARTICLE_PATTERNS = [
    (re.compile(r"\b(Article|Art\.)\s*(\d+[A-Za-z\-]*)", re.I), "Art. {num}"),
    (re.compile(r"\bRecital\s*(\d+)", re.I), "Recital {num}"),
    (re.compile(r"\bKapittel\s*(\d+)", re.I), "Kapittel {num}"),
    (re.compile(r"\b§\s*(\d+[a-zA-Z]*)", re.I), "§ {num}"),
]

def guess_article_label(text: str) -> str:
    head = "\n".join(text.splitlines()[:4])
    for rx, fmt in ARTICLE_PATTERNS:
        m = rx.search(head)
        if m:
            return fmt.format(num=m.group(1))
    return ""

# --------------------------- Fetching & Extraction ---------------------------

def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"

def _fetch(url: str, accept_pdf_first: bool = True, timeout: int = 45):
    """
    Fetch a URL with browser-like headers.
    Returns (bytes, content_type, final_url).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/pdf, text/html;q=0.9,*/*;q=0.8" if accept_pdf_first else "*/*",
        "Accept-Language": "en;q=0.9,no;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return resp.content, ctype, resp.url

def _find_pdf_href_in_html(html: str, base_url: str) -> str | None:
    """
    Scan HTML for a likely PDF link.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) <a> or <link> with .pdf or type=application/pdf
    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href") or ""
        typ  = (tag.get("type") or "").lower()
        if href.lower().endswith(".pdf") or "application/pdf" in typ:
            return requests.compat.urljoin(base_url, href)

    # 2) Regex fallback
    m = re.search(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', html, flags=re.I)
    if m:
        return requests.compat.urljoin(base_url, m.group(1))

    return None

def fetch_pdf_or_html(url: str) -> Dict[str, Any]:
    """
    Try hard to retrieve real PDF bytes; otherwise return HTML.
    Returns: {"kind": "pdf"|"html", "data": bytes|str, "final_url": str}
    """
    data, ctype, final_url = _fetch(url, accept_pdf_first=True)

    # Case 1: Direct PDF by header or magic
    if "application/pdf" in ctype or _looks_like_pdf(data):
        return {"kind": "pdf", "data": data, "final_url": final_url}

    # Case 2: HTML wrapper; try to find nested PDF href
    try:
        html = data.decode("utf-8", errors="ignore")
    except Exception:
        html = ""

    pdf_link = _find_pdf_href_in_html(html, final_url)
    if pdf_link:
        data2, ctype2, final_url2 = _fetch(pdf_link, accept_pdf_first=True)
        if "application/pdf" in ctype2 or _looks_like_pdf(data2):
            return {"kind": "pdf", "data": data2, "final_url": final_url2}

    # Case 3: Fall back to HTML text
    return {"kind": "html", "data": html or data.decode("latin-1", errors="ignore"), "final_url": final_url}

def save_bytes(path: pathlib.Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def read_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes via pypdf. Returns empty string on failure.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"[warn] PDF parse failed; returning empty text. Reason: {e}")
        return ""

def read_html_text(url: str) -> str:
    data, _, final_url = _fetch(url, accept_pdf_first=False)
    html = data.decode("utf-8", errors="ignore")
    return read_html_text_from_string(html, base_url=final_url)

def read_html_text_from_string(html: str, base_url: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --------------------------- Source Normalization ---------------------------

def _normalize_source(src: Any) -> Dict[str, Any]:
    """
    Accept a dict (preferred) or a plain string (URL/path).
    Ensures the rest of the pipeline has the keys it expects.
    """
    if isinstance(src, str):
        url = src
        guess_title = (url.rsplit("/", 1)[-1] or url)
        is_pdf = url.lower().endswith(".pdf")
        return {
            "title": guess_title,
            "act": guess_title,
            "jurisdiction": "",
            "language": "",
            "url": url,
            "download": url if is_pdf else None,
            "version_date": "",
            "doc_type": ""
        }
    # dict-like: make sure keys exist
    return {
        "title": src.get("title") or src.get("act") or (src.get("url") or ""),
        "act": src.get("act") or src.get("title") or "",
        "jurisdiction": src.get("jurisdiction", ""),
        "language": src.get("language", ""),
        "url": src.get("url") or "",
        "download": src.get("download"),
        "version_date": src.get("version_date", ""),
        "doc_type": src.get("doc_type", ""),
    }

def _iter_sources(sources_json: Any) -> Iterable[Dict[str, Any]]:
    """
    Yield normalized sources from either:
    - a flat list, or
    - a dict of lists (e.g., {"regulations":[...], "acts_and_laws":[...]})
    - a single item (dict or string)
    """
    if isinstance(sources_json, list):
        for item in sources_json:
            yield _normalize_source(item)
    elif isinstance(sources_json, dict):
        for _group, items in sources_json.items():
            if isinstance(items, list):
                for item in items:
                    yield _normalize_source(item)
            else:
                yield _normalize_source(items)
    else:
        yield _normalize_source(sources_json)

# --------------------------- Index Builder ---------------------------

def build_index():
    sources_path = ROOT / "sources.json"
    if not sources_path.exists():
        raise FileNotFoundError(f"Could not find sources.json at {sources_path}")

    sources_json = json.loads(sources_path.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    coll = client.get_or_create_collection(
        name="legal_no_eu",
        metadata={"hnsw:space": "cosine", "version": "v1"},
        embedding_function=None,  # we pass embeddings explicitly
    )

    records: List[Dict[str, Any]] = []

    for src in _iter_sources(sources_json):
        text = ""
        checksum_source_bytes = b""

        # Pick a primary URL; prefer 'download' if present (usually PDF)
        primary = src.get("download") or src.get("url")
        if not primary:
            print(f"[warn] No URL/download for {src.get('title')}; skipping.")
            continue

        fetch_res = fetch_pdf_or_html(primary)

        if fetch_res["kind"] == "pdf":
            pdf_bytes = fetch_res["data"]
            checksum_source_bytes = pdf_bytes

            # Save a copy for provenance / debugging
            fname_hint = src["act"] or src["title"] or "document"
            fname = re.sub(r"[^A-Za-z0-9\-\.]+", "_", fname_hint) + ".pdf"
            out_path = DOCS / fname
            save_bytes(out_path, pdf_bytes)

            text = read_pdf_text_from_bytes(pdf_bytes)

            # If the PDF text is empty (scanned or image-only), attempt HTML extraction from 'url' if different.
            if not text.strip() and src.get("url") and src.get("url") != primary:
                print(f"[info] Empty PDF text for {fname}; trying HTML extraction from {src['url']}")
                html_res = fetch_pdf_or_html(src["url"])
                if html_res["kind"] == "html":
                    text = read_html_text_from_string(html_res["data"], base_url=html_res["final_url"])

        else:
            # HTML route (wrapper or plain page)
            html = fetch_res["data"]
            checksum_source_bytes = html.encode("utf-8", errors="ignore")
            text = read_html_text_from_string(html, base_url=fetch_res["final_url"])

        if not text.strip():
            print(f"[warn] No extractable text for {src['title']} ({primary}); skipping.")
            continue

        chksum = checksum_bytes(checksum_source_bytes)

        # Split & chunk
        for block in split_heading_aware(text):
            for piece in semantic_chunk(block, target_tokens=800, overlap_tokens=100):
                article = guess_article_label(piece)
                records.append({
                    "text": piece,
                    "meta": {
                        "title": src["title"],
                        "jurisdiction": src["jurisdiction"],
                        "act": src["act"],
                        "article": article,
                        "url": src["url"],
                        "version_date": src["version_date"],
                        "language": src["language"],
                        "checksum": chksum
                    }
                })

    if not records:
        print("[error] No records prepared; aborting index write.")
        return

    # Embed & write
    print(f"[info] Embedding {len(records)} chunks with {EMB_MODEL_NAME} ...")
    docs = [r["text"] for r in records]
    embeds = EMBEDDER.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    ids = [f'{rec["meta"]["act"]}-{i:07d}' for i, rec in enumerate(records)]

    coll.add(
        ids=ids,
        documents=docs,
        metadatas=[r["meta"] for r in records],
        embeddings=[e.tolist() for e in embeds]
    )
    print(f"Indexed {len(records)} chunks into {INDEX_DIR.absolute()}")

# --------------------------- Entrypoint ---------------------------

if __name__ == "__main__":
    build_index()
