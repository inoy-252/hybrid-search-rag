# 🚀 Session Context & Continuing Roadmap

## 👤 Profile & Background
* **Developer:** Yasir Lone (Srinagar, J&K, India)
* **Background:** Self-taught AI/ML Engineer transitioning from 12th Medical (PCB). Highly dedicated, learning from first principles to outshine traditional CS degree holders.
* **Goal:** Become an elite AI/LLM Systems Engineer. Applied for the LLM Engineer role at AlgoProfessor AI R&D with a project-backed resume and cold pitch.

---

## 🏗️ What Has Been Built (In Repository: `30-days-of-python-practice`)
1. **Core Python Foundations:** Completed all 30 days (Advanced OOP, SQLite, Concurrency, APIs).
2. **Autonomous Market Intelligence Pipeline (`projects/market_pipeline/`):** Multi-stage resilient ingestion, NLP keyword extraction, financial metrics, and multi-format persistence.
3. **Hybrid Search RAG Pipeline from Scratch (`projects/hybrid_search_rag/`):**
   * `data.py`: 5 production engineering runbooks (Auth, DB Failover, Infra Auto-scaling, Payments, OpenTelemetry Observability).
   * `bm25.py`: Okapi BM25 implementation from scratch (tokenization, IDF, term frequency saturation with $k_1$ and $b$).
   * `dense_retriever.py`: Google `gemini-embedding-2` 3,072-dimensional vector generator + manual NumPy cosine similarity.
   * `fusion.py`: Reciprocal Rank Fusion (RRF) algorithm to normalize disparate score scales.
   * `reranker.py`: LLM cross-encoder judge with `gemini-3.5-flash-lite` scoring passages 0.0–10.0.
   * `pipeline.py`: Interactive terminal CLI combining the entire end-to-end pipeline.

---

## 🎯 Current Task: Build the Query Intelligence Layer (`query_intelligence.py`)

### The Problem:
* **Vocabulary Mismatch & Query Asymmetry:** Casual user queries (e.g. *"how do we stop duplicate charges when someone clicks pay twice"*) fail keyword search because runbooks use domain terminology (e.g. *"idempotency key"*, *"ERR-PAY-409-DUPLICATE"*).
* Dense embeddings also suffer from query-document structural mismatch (short vague questions vs. long authoritative specs).

### What We Are Building:
Create `projects/hybrid_search_rag/query_intelligence.py` containing a `QueryIntelligence` class with:
1. **Query Rewriting / Expansion (`rewrite_query`):** Translates conversational user complaints into technical engineering terminology and keywords optimized for BM25.
2. **HyDE (`generate_hypothetical_document`):** Generates a hypothetical technical runbook paragraph so its embedding closely aligns in vector space with the true document vectors.
3. Integration with `pipeline.py`.

---

## ⚠️ Important Pedagogical Rules for the Assistant:
* **DO NOT just dump finished code.** 
* Yasir wants to learn every concept deeply and write the implementation himself.
* Guide him step-by-step, ask him questions, explain the math/theory, and review his code as he writes it.
