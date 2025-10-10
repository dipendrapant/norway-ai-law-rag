import html
import json
import os
import pathlib
import re
import time
from typing import Any, Dict, List, Tuple

import chromadb
import gradio as gr
import numpy as np
import requests
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

gr.set_static_paths([])

ROOT = pathlib.Path(__file__).parent
INDEX_DIR = ROOT / "index"

DEFAULT_SCOPE = os.getenv("DEFAULT_SCOPE", "both")  # eu|no|both
BUILD_INDEX = os.getenv("BUILD_INDEX_ON_START", "false").lower() == "true"
POLICY_URL = os.getenv("POLICY_URL")

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
  

SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+|\n+')

# Build index if empty (optional for local)
if BUILD_INDEX and not (INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT/"index.py")], check=True)

# ------- Embeddings / Vector store -------
EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
client = chromadb.PersistentClient(path=str(INDEX_DIR))
coll = client.get_or_create_collection(name="legal_no_eu", embedding_function=None)
INDEX_VERSION = (coll.metadata or {}).get("version","v1")

# ------- Guardrails -------
CLINICAL_TERMS = r"(diagnos|treat|symptom|therapy|dose|dosage|medication|prescrib|fracture|tumou?r|cancer|diabetes|blood pressure|ECG|MRI|CT|lab result)"
IDENTIFIERS = r"(\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b\d{11}\b)"
EMAIL = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
PHONE = r"\b(\+?\d[\d\-\s]{6,}\d)\b"
DOB = r"\b(19|20)\d{2}[-/.]\d{2}[-/.]\d{2}\b"

def _top_sentence_by_embedding(query: str, text: str) -> tuple[str, float]:
    """
    Pick the most relevant sentence/short line from text for the query.
    """
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and len(p.strip()) > 2]
    if not parts:
        return text.strip()[:300], 0.0
    qv = EMBEDDER.encode([query], normalize_embeddings=True)[0]
    sv = EMBEDDER.encode(parts, normalize_embeddings=True)
    scores = np.dot(sv, qv)  # cosine since normalized
    idx = int(np.argmax(scores))
    return parts[idx][:500], float(scores[idx])

ART_RX = re.compile(
    r'\b(Article|Art\.)\s*(\d+[A-Za-z\-]*)|\bRecital\s*(\d+)|\bKapittel\s*(\d+)|§\s*([0-9A-Za-z]+)',
    re.I
)

def _guess_article_label(text: str) -> str:
    head = "\n".join(text.splitlines()[:6])
    m = ART_RX.search(head) or ART_RX.search(text)
    if not m:
        return ""
    g = m.groups()
    if g[1]: return f"Art. {g[1]}"       # Article / Art.
    if g[2]: return f"Recital {g[2]}"    # Recital
    if g[3]: return f"Kapittel {g[3]}"   # Kapittel
    if g[4]: return f"§ {g[4]}"          # Paragraph sign
    return ""
    

def _rank_sentences(query: str, text: str):
    """Return (sentences, scores, embeddings) for all sentences."""
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p and len(p.strip()) > 2]
    if not parts:
        return [], np.array([]), np.empty((0, EMBEDDER.get_sentence_embedding_dimension()))
    qv = EMBEDDER.encode([query], normalize_embeddings=True)[0]
    sv = EMBEDDER.encode(parts, normalize_embeddings=True)
    scores = sv @ qv  # cosine because normalized
    return parts, scores, sv

def _top_k_diverse(query: str, text: str, k: int = 3, redundancy_thresh: float = 0.85):
    """
    Pick up to k strong, non-duplicate sentences from text.
    Deduplicate by cosine similarity between sentence embeddings.
    """
    sents, scores, sv = _rank_sentences(query, text)
    if len(sents) == 0:
        return []
    order = np.argsort(-scores)
    chosen = []
    chosen_vecs = []
    for idx in order:
        cand = sents[idx]
        vec = sv[idx]
        if not chosen_vecs:
            chosen.append((cand, float(scores[idx])))
            chosen_vecs.append(vec)
        else:
            sims = np.dot(np.vstack(chosen_vecs), vec)
            if np.max(sims) < redundancy_thresh:
                chosen.append((cand, float(scores[idx])))
                chosen_vecs.append(vec)
        if len(chosen) >= k:
            break

    return [c for c, _ in chosen]


def _show_spinner():
    return spinner_html.replace('display:none', 'display:flex')

def _hide_spinner():
    return spinner_html  # back to display:none

def contains_phi(text: str) -> bool:
    t = text.lower()
    risky = re.search(CLINICAL_TERMS, t)
    id_like = re.search(IDENTIFIERS, text) or re.search(EMAIL, text) or re.search(PHONE, text) or re.search(DOB, text)
    return bool(risky or id_like)

def scrub_pii(q: str) -> str:
    q = re.sub(EMAIL, "[EMAIL]", q)
    q = re.sub(PHONE, "[PHONE]", q)
    q = re.sub(DOB, "[DATE]", q)
    return q.strip()

def in_scope(q: str) -> bool:
    keywords = ["gdpr","data","health","journal","record","ai","risk","consent","research","ehds","normen","helseregister","helse","pasient","personvern","privacy","data protection"]
    return any(k in q.lower() for k in keywords)

# ------- Retrieval -------
def bm25_rerank(query: str, docs: List[Dict[str,Any]], top_n=8) -> List[Dict[str,Any]]:
    corpus = [d["document"] for d in docs]
    tok_corpus = [c.split() for c in corpus]
    bm25 = BM25Okapi(tok_corpus)
    scores = bm25.get_scores(query.split())
    order = np.argsort(-scores)[:top_n]
    return [docs[i] for i in order]

def format_citation(meta: Dict[str,str]) -> str:
    art = meta.get("article") or ""
    act = meta.get("act") or meta.get("title")
    return f"[{act} {art}]" if art else f"[{act}]"

def build_context(chunks: List[Dict[str,Any]], budget_chars=7000) -> Tuple[str, List[Dict[str,Any]]]:
    pieces, used, cur = [], [], 0
    for d in chunks:
        cite = format_citation(d["metadata"])
        line = f"{cite} — {d['metadata']['url']}\n{d['document']}\n"
        if cur + len(line) > budget_chars: break
        pieces.append(line); used.append(d); cur += len(line)
    return "\n---\n".join(pieces), used

def search(query: str, scope: str, k: int = 8) -> List[Dict[str,Any]]:
    q_clean = scrub_pii(query)
    q_vec = EMBEDDER.encode([q_clean], normalize_embeddings=True)[0].tolist()
    where = None
    if scope == "eu": where = {"jurisdiction": "EU"}
    elif scope == "no": where = {"jurisdiction": "NO"}

    res = coll.query(query_embeddings=[q_vec], n_results=k, where=where,
                     include=["documents","metadatas","embeddings"])
    docs = []
    for i in range(len(res["documents"][0])):
        docs.append({"id": res["ids"][0][i],
                     "document": res["documents"][0][i],
                     "metadata": res["metadatas"][0][i]})
    return bm25_rerank(q_clean, docs, top_n=k)

# ------- Generator (Ollama) -------
SYS_PROMPT = """You are a compliance assistant for health informatics in Norway/EU.
Answer ONLY from the provided context below. Quote article/section numbers and return inline bracket citations like [GDPR Art. 9(2)(j)] or [Health Personnel Act §29].
If context is insufficient or off-topic, reply exactly: "Not enough information in sources to answer reliably."
Do NOT provide clinical advice; summarize regulatory aspects only.
Reply in the same language as the user’s question.
"""


def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=1.5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models",[])]
    except Exception:
        return []

def ollama_chat(model: str, system: str, user: str) -> str:
    """
    Call Ollama chat endpoint. Prefer non-streaming for simpler JSON parsing.
    Falls back to streaming parser if the server still streams.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.1},
        "stream": False,  
    }
    headers = {"Accept": "application/json"}

    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {}) or {}
        return (msg.get("content") or "").strip()
    except requests.exceptions.JSONDecodeError:
        pass

    try:
        payload["stream"] = True
        with requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, headers=headers, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            content_parts = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                delta = ""
                if isinstance(obj, dict):
                    if "message" in obj and isinstance(obj["message"], dict):
                        delta = obj["message"].get("content", "") or ""
                    elif "response" in obj:  
                        delta = obj.get("response", "") or ""
                if delta:
                    content_parts.append(delta)
            return "".join(content_parts).strip()
    except Exception as e:
        return f"Not enough information in sources to answer reliably. [engine error: {e}]"


def ensure_citations(answer: str, used: List[Dict[str,Any]]) -> str:
    if re.search(r"\[[^\]]+\]", answer): 
        return answer
    first = used[0]["metadata"] if used else None
    if first:
        cite = format_citation(first)
        lines = [ln.strip() for ln in answer.split("\n") if ln.strip()]
        return "\n".join(f"{ln} {cite}" for ln in lines)
    return answer

def verify(answer: str, used: List[Dict[str,Any]]):
    if "Not enough information" in answer:
        return False, ["Model abstained due to low context sufficiency."]
    valid = set(format_citation(d["metadata"]) for d in used)
    problems = []
    for s in re.split(r"(?<=[\.\?\!])\s+", answer.strip()):
        if not s: continue
        cites = re.findall(r"\[[^\]]+\]", s)
        if not cites or not any(c in valid for c in cites):
            problems.append(f"Missing/invalid citation for: {s[:80]}...")
    return (len(problems)==0), problems

def generate_answer(query: str, context: str, model_choice: str) -> str:
    user_prompt = f"User question:\n{query}\n\nContext:\n{context}\n\nProvide the answer now."
    return ollama_chat(model_choice, SYS_PROMPT, user_prompt) if USE_OLLAMA \
           else "Not enough information in sources to answer reliably."

def answer_api(query: str, scope: str, consent: bool, model_choice: str) -> Dict[str,Any]:
    t0 = time.time()
    if not consent:
        return {"ok": False, "answer": "Please confirm the transparency/consent notice to proceed.", "citations": []}
    if contains_phi(query):
        return {"ok": False, "answer": "Your question appears to include personal or health-identifying information or clinical advice. Please remove PHI and avoid clinical questions.", "citations": []}
    if not in_scope(query):
        return {"ok": False, "answer": "Out of scope. This assistant covers EU/NO health data, AI, and related compliance only.", "citations": []}

    docs = search(query, scope, k=8)
    context, used = build_context(docs, budget_chars=7000)
    if not used:
        return {"ok": False, "answer": "Not enough information in sources to answer reliably.", "citations": []}

    draft = ensure_citations(generate_answer(query, context, model_choice), used)
    ok, _ = verify(draft, used)
    if not ok:
        #return {"ok": False, "answer": "Not enough information in sources to answer reliably.", "citations": []}
        draft = ensure_citations(draft, used)

    per_doc = {}  
    for d in used:
        meta = d["metadata"]
        doc_text = d["document"]
        act = (meta.get("act") or meta.get("title") or "").strip()
        url = (meta.get("url") or "").strip()
        key = (act, url)
        lines = _top_k_diverse(query, doc_text, k=3, redundancy_thresh=0.85)

        article = (meta.get("article") or "").strip()
        entry = per_doc.get(key)
        if not entry:
            per_doc[key] = {
                "act": act,
                "url": url,
                "articles": set([article] if article else []),
                "lines": lines[:]  # copy
            }
        else:
            if article:
                entry["articles"].add(article)
            seen = set(entry["lines"])
            for ln in lines:
                if ln not in seen:
                    entry["lines"].append(ln)
                    seen.add(ln)

    sources = []
    for (act, url), info in per_doc.items():
        pin_lines = info["lines"][:5] if info["lines"] else []
        title_multiline = "<br>".join(f"• {ln}" for ln in pin_lines) if pin_lines else ""
        art_label = ", ".join(sorted(a for a in info["articles"] if a))
        citation = f"{act} {art_label}".strip()
        sources.append({
            "citation": citation,
            "title": title_multiline,  # contains <br> for separate lines
            "url": info["url"]
        })

    # --- Log and return ---
    # latency_ms = int((time.time() - t0) * 1000)
    # log_event({
    #     "q_hash": hash_query(query),
    #     "scope": scope,
    #     "index_version": INDEX_VERSION,
    #     "doc_ids": [d["id"] for d in used],
    #     "ok": True,
    #     "latency_ms": latency_ms,
    #     "model": model_choice or DEFAULT_OLLAMA_MODEL
    # })
    # purge_old_logs()
    return {"ok": True, "answer": draft, "citations": sources}


# ------- UI -------
EXAMPLES = [
  "Is de-identification enough for secondary use of patient data for research?",
  "What does EHDS change for cross-border secondary use of health data?",
  "Who may access patient records within an organization?",
  "Which AI system categories are prohibited in healthcare?",
  "Can we process health data without consent for research under GDPR?"
]

def ui_answer(q, scope, consent, model_choice):
    res = answer_api(q, scope, consent, model_choice)
    if not res["ok"]:
        return res["answer"], ""

    lines = [
        "| **Reference** | **Title/Sentence in reference document** | **URL** |",
        "|:--|:--|:--|"
    ]
    for s in res["citations"]:
        citation = (s.get("citation","") or "").replace("|", "\\|").strip()
        title    = (s.get("title","") or "").strip()
        url      = (s.get("url","") or "").strip()
        url_md   = f"[{url}]({url})" if url else ""
        lines.append(f"| {citation} | {title} | {url_md} |")

    sources_md = "\n".join(lines)
    return res["answer"], sources_md



EU_BADGE = "HuggingFace can Hosted it in any region. ⚠️ Check Space settings → choose an EU region if you want." 
ollama_models = list_ollama_models() if USE_OLLAMA else []
if USE_OLLAMA:
    base_choices = set(ollama_models) | {DEFAULT_OLLAMA_MODEL, "llama3:latest",
                                         "qwen2.5:7b-instruct", "qwen2.5:7b-instruct-q4_K_M"}
    ollama_models = sorted(base_choices)

with gr.Blocks(title="NO/EU Health Compliance RAG") as demo:
    gr.HTML("""
    <style>
    .prose table { table-layout: fixed; width: 100%; }
    .prose th:nth-child(1), .prose td:nth-child(1) { width: 15%; }
    .prose th:nth-child(2), .prose td:nth-child(2) { width: 60%; white-space: normal; word-wrap: break-word; }
    .prose th:nth-child(3), .prose td:nth-child(3) { width: 25%; white-space: normal; word-wrap: break-word; }
    </style>
    """)

    gr.Markdown(
        f"""
# 🇳🇴🇪🇺 AI and Health Informatics Related Compliance RAG for the Context of Norway: Education Purpose Only

**Transparency:** This is an AI system (a RAG) that generates answers with reference and corresponding sentence or text in the documents. The knowledge index is created using document embeddings derived from the from EU/NO legal docuements, read [about project]({POLICY_URL}/limitations.md). It is **not legal or clinical advice**.  
Read the [Privacy Notice]({POLICY_URL}/privacynotice.md). |&nbsp; **Hosting:** {EU_BADGE}
""")
    with gr.Row():
        with gr.Column(scale=2):
            consent = gr.Checkbox(label="I understand this is an AI system and I have read the Privacy Notice", value=False)
            query = gr.Textbox(label="Your question", placeholder="e.g., Can we use patient data for AI model validation without consent?")
            scope = gr.Radio(choices=["both","eu","no"], value=DEFAULT_SCOPE, label="Scope filter")
            model_choice = gr.Dropdown(choices=ollama_models, value=DEFAULT_OLLAMA_MODEL,
                                       allow_custom_value=True, label="Local model (Ollama)")
            ask = gr.Button("Ask")
            gr.Examples(EXAMPLES, [query], examples_per_page=5)
        with gr.Column(scale=3):
            out = gr.Markdown(label="Answer")
            src = gr.Markdown(label="Sources")

                # lightweight spinner (hidden by default)
            spinner_html = """
                                <div id="spinner" style="
                                display:none;
                                position:fixed;
                                top:0;
                                left:0;
                                width:100vw;
                                height:100vh;
                                background:rgba(255,255,255,0.8);
                                z-index:9999;
                                justify-content:center;
                                align-items:center;
                                flex-direction:column;
                                gap:12px;
                                font-size:1.1rem;
                                ">
                                <div style="
                                    width:40px;
                                    height:40px;
                                    border:4px solid rgba(0,0,0,0.1);
                                    border-top-color:#007bff;
                                    border-radius:50%;
                                    animation:spin 0.9s ease-in-out infinite;
                                    box-shadow:0 0 8px rgba(0,0,0,0.2);
                                "></div>
                                <span style="font-family:Arial, sans-serif; color:#333;">I am searching....</span>
                                </div>

                                <style>
                                @keyframes spin {
                                to { transform: rotate(360deg); }
                                }
                                </style> """
            spinner = gr.HTML(spinner_html)

    ask.click(_show_spinner, inputs=[], outputs=[spinner], queue=False).then(ui_answer, inputs=[query, scope, consent, model_choice],outputs=[out, src], show_progress="hidden").then(_hide_spinner, inputs=[], outputs=[spinner], queue=False)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=8501)
