This is a lightweight template for building Retrieval-Augmented Generation (RAG) applications. It follows this:
**_List all the documents and URLs to consider → Create Knowledge Index (see index.py) → User → Query → Vector Store → Document Retrieval → Answer + With Source Reference_**

#### 1) Create venv & install deps

`python -m venv your_venv_name`
`source your_venv_name/bin/activate`
`pip install -r requirements.txt`

#### 2) Configure env

.env.example

#### 3) Build index (downloads PDFs once, parses, embeds)

`python index.py`

#### 4) Run

`python server.py`

It is running at: `https://huggingface.co/spaces/pantdipendra/uit-norway-ai-law` with the same knowledge index
