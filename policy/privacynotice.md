# Privacy Notice (Demo RAG on Hugging Face)

**Purpose & legal basis.**  
This demo is for teaching/learning and exploratory research on legal-text retrieval. Processing relies on **legitimate interests** in education and system evaluation. We also ask you to **consent** (checkbox) before answering, to promote transparency and safe use.

**What we process.**

- **Your input:** the question text and UI options (e.g., scope/model).
- **Transient use only:** your raw question is used **in memory** to retrieve passages and draft an answer.
- **We do not store raw questions.**
- **Minimal logs:** we store only a **salted hash** of the query plus minimal technical metadata (doc IDs, latency, model name, index version) to audit performance and detect misuse. The salted hash cannot be reversed to recover your text.

**Do not submit sensitive data.**  
Please **do not include** personal health information (PHI), names, national IDs, emails, phone numbers, dates of birth, or other identifiers. We run basic checks and may reject such inputs, but you are responsible for what you type.

**Retention.**  
Hashed logs and minimal metadata are **auto-deleted after 30 days**.

**Hosting & transfers.**  
The app runs on **Hugging Face Spaces** infrastructure. We **aim** to deploy in an **EU region**, but the actual region may vary by Space settings. If the banner indicates a non-EU region, treat it as an international transfer and **do not use** the demo for sensitive content.

**Recipients / processors.**

- **Hugging Face** hosts the Space (infrastructure provider).
- The local LLM runs **inside** the Space container via **Ollama**; prompts and retrieved text are processed locally within that runtime.
- We do **not** share your data with third parties for marketing or profiling.

**Cookies / analytics.**  
We do not intentionally add analytics scripts. Hugging Face may set necessary cookies to operate Spaces; see their policies.

**Security.**

- Transport is over HTTPS (as provided by Hugging Face).
- We avoid storing raw questions and keep only salted hashes w/ minimal metadata.
- Access to the Space and logs is restricted to project maintainers.

**Your choices & rights.**

- You can use the app **without entering personal data**.
- Because we do not store raw questions and keep only salted hashes, we generally **cannot identify you** to fulfill access/erasure requests on specific texts.
- For inquiries, contact **PRIVACY_CONTACT** shown in the app banner.

**Children.**  
Not intended for use by children. Do not submit any personal information about minors.

**Changes.**  
We may update this notice as we change sources, models, or hosting. Material changes will be reflected here.

---

**Summary:** This demo RAG processes your question **transiently**, stores only **salted hashes** with minimal technical metadata for up to **30 days**, and should **not** be used for sensitive or identifying information. Always verify results against the **linked legal sources**.
