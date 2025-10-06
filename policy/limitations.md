## What this system

- **Purpose:** A Retrieval-Augmented Generation (RAG) assistant focused on EU/NO health-data and AI compliance (e.g., GDPR, AI Act, EHDS, Norwegian acts/regulations and Normen).
- **How it works:**
  1. We **index** a curated set of legal sources from `sources.json` into a local Chroma database using a **multilingual** embedding model (`paraphrase-multilingual-mpnet-base-v2`).
  2. At query time, we **retrieve** the most relevant passages and **pinpoint** one or more highly-relevant **sentences/paragraphs** inside each document.
  3. A local LLM (via **Ollama**, default **llama3**) **summarizes** strictly from those retrieved passages.
  4. We display **inline bracket citations** in the answer, and a **Sources** panel showing the **exact sentence(s)** per document with a **clickable URL**.

> This system is meant to help you navigate the texts; it does **not** replace professional legal review.

---

## limitations

### 1) Not legal or clinical advice

- Outputs are **informational** and **educational** only.
- The assistant is **not** legal counsel and **not** a medical device; it must not be used to make legal or clinical decisions.

### 2) Coverage is limited to curated sources

- Answers are constrained to the documents listed in `sources.json`.
- If a topic or clause isn’t in those sources (or in the indexed version), the system will abstain (“Not enough information…”).

### 3) Freshness / version drift

- Laws and guidance change. The index reflects the **point-in-time** copies we downloaded.
- New amendments, corrigenda, consolidated versions, or guidance published **after indexing** will **not** be reflected until you **rebuild the index**.

### 4) Retrieval and parsing constraints

- Some official “PDF” links serve HTML wrappers; we follow links best-effort.
- **Scanned image-only PDFs are not OCR-processed**; text may be missing or partial.
- Headings and article/section detection are heuristic; **pinpoint citations** (Art./Recital/§/Kapittel) aim to be precise but can occasionally be off by a heading level or nearby paragraph.

### 5) Generation reliability

- The LLM is instructed to answer **only from provided context** and include citations, but **hallucinations remain possible**.
- We try to enforce citations and show a **Sources** table with the **exact sentence(s)** and **clickable URLs** for verification.

### 6) Language and translation nuances

- The system supports **English and Norwegian** input/output.
- When reading Norwegian laws or unofficial translations, subtle **legal nuances** can be lost in translation. Always verify against the official text.

### 7) Scope & safety filters

- Out-of-scope questions (unrelated to EU/NO health-data/AI compliance) are declined.
- Personal/health identifiers trigger a safety block; the assistant is **not** for case-specific or patient-identifying queries.

### 8) Performance & hosting

- Response time depends on local model size and Hugging Face Space resources.
- On free/shared runtimes, there may be cold starts, timeouts, or rate limits.

---

## Transparency reminder

- **Always verify** statements against the **linked legal texts**.
- If the answer lacks sufficient citations or context, treat it as **non-authoritative** and consult the source directly.
