# 🚀 Production Hybrid Search RAG Pipeline (From Scratch)

A production-grade, zero-framework **Hybrid Search & Retrieval-Augmented Generation (RAG)** pipeline implemented from first principles in Python.

This system combines sparse lexical retrieval (Okapi BM25), dense semantic retrieval (Google Gemini 3,072-dimensional embeddings), Reciprocal Rank Fusion (RRF), an LLM cross-encoder reranker, and a dual-path **Query Intelligence Layer (Query Rewriting + HyDE)** to eliminate hallucinations and solve vocabulary mismatch in complex engineering runbooks.

---

## 🏗️ Architecture & Pipeline Flow

```text
                               [ User Query ]
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       Query Intelligence Engine (Gemini)         │
             └──────────────────────────────────────────────────┘
                      │                                │
        [1. Query Rewriting]             [2. HyDE (Hypothetical Doc)]
      (Expands technical keywords)       (Predicts runbook structure)
                      │                                │
                      ▼                                ▼
         ┌─────────────────────────┐      ┌─────────────────────────┐
         │  Sparse Search (BM25)   │      │ Dense Search (Vectors)  │
         │ (Okapi BM25 from scratch)│     │(Gemini 3072-d Embeddings)│
         └─────────────────────────┘      └─────────────────────────┘
                      │                                │
                      └───────────────┬────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │ Reciprocal Rank Fusion (RRF)   │
                      │  (Eliminates scale mismatch)   │
                      └────────────────────────────────┘
                                      │
                                      ▼ [Top Candidates]
                      ┌────────────────────────────────┐
                      │   LLM Cross-Encoder Reranker   │
                      │   (Pointwise 0.0–10.0 Scoring) │
                      └────────────────────────────────┘
                                      │
                                      ▼ [Top Grounded Document]
                      ┌────────────────────────────────┐
                      │ Grounded Generator & Synthesis │
                      │  (Zero-hallucination answers)  │
                      └────────────────────────────────┘
```

---

## 🧩 Core Hand-Crafted Components

1. **Query Intelligence (`query_intelligence.py`)**:
   * **Query Rewriting:** Translates informal, conversational user questions into high-density technical keywords and error codes for BM25.
   * **HyDE (Hypothetical Document Embeddings):** Hallucinates a realistic technical runbook excerpt to align vector geometry before dense semantic search.

2. **Sparse Lexical Retrieval (`bm25.py`)**:
   * Pure Python implementation of **Okapi BM25** from scratch without external IR packages.
   * Features custom tokenization, Inverse Document Frequency (IDF), and term frequency saturation with tunable parameters ($k_1 = 1.5, b = 0.75$).

3. **Dense Semantic Retrieval (`dense_retriever.py`)**:
   * Generates 3,072-dimensional vector embeddings using Google's `gemini-embedding-2`.
   * Computes Cosine Similarity manually using **NumPy** matrix operations.

4. **Fusion Layer (`fusion.py`)**:
   * Implements **Reciprocal Rank Fusion (RRF)**:
     $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
   * Merges disparate score distributions purely by rank position ($k = 60$), eliminating score-scale disparities between lexical and semantic engines.

5. **Cross-Encoder Reranker (`reranker.py`)**:
   * Employs `gemini-3.5-flash-lite` as a point-wise relevance judge.
   * Inspects candidate passages side-by-side with the original user query and assigns a calibrated relevance score (0.0 to 10.0) to produce high-precision final rankings.

6. **Grounded Answer Generator (`pipeline.py`)**:
   * Feeds top-ranked context into Gemini with strict grounding constraints to guarantee zero hallucinations.

---

## ⚡ Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/inoy-252/hybrid-search-rag.git
cd hybrid-search-rag
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key
Create a `.env` file in the root directory:
```text
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey)).*

### 5. Run the Interactive Pipeline
```bash
python pipeline.py
```

---

## 📁 Repository Structure

```text
├── bm25.py                # Okapi BM25 implementation from scratch
├── dense_retriever.py     # Gemini 3072-d embeddings & NumPy cosine similarity
├── fusion.py              # Reciprocal Rank Fusion (RRF) algorithm
├── reranker.py            # LLM Cross-Encoder relevance judge
├── query_intelligence.py  # Query Rewriting & HyDE implementation
├── pipeline.py            # End-to-end interactive terminal application
├── data.py                # Internal engineering runbook corpus
├── requirements.txt       # Minimal dependency requirements
└── README.md              # Documentation & Architecture breakdown
```

---

## 📜 License
MIT License. Free to use and build upon.
