from fastapi import FastAPI
from pydantic import BaseModel

from bm25 import BM25
from data import DOCUMENTS
from dense_retriever import DenseRetriever
from fusion import reciprocal_rank_fusion
from query_intelligence import QueryIntelligence
from reranker import Reranker

app = FastAPI(
    title="Production Hybrid Search RAG API",
    description="Zero-framework Hybrid Search and RAG Engine powered by BM25, Gemini Vectors, RRF, and HyDE.",
    version="1.0.0",
)

print("Starting up Hybrid RAG engines.")
bm25 = BM25(DOCUMENTS)
dense = DenseRetriever(DOCUMENTS)
qi = QueryIntelligence()
reranker = Reranker()
print("All RAG engines ready!\n")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str
    results: list


class ChatResponse(BaseModel):
    query: str
    answer: str
    source_document_id: str
    source_title: str
    relevance_score: float


def generate_grounded_answer(query: str, document: dict) -> str:
    prompt = f"""You are a helpful engineering assistant. 
Answer the engineer's question using ONLY the information in the document below.
If the document does not contain the answer, say "This document does not cover that."
Engineer's Question: {query}
Document Title: {document["title"]}
Document Content:
{document["content"]}
Provide a clear, direct, actionable answer."""
    return qi._generate_with_retry(prompt)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Hybrid Search RAG"}


@app.post("/search", response_model=SearchResponse)
def search_documents(request: SearchRequest):

    rewritten_query = qi.rewrite_query(request.query)
    hyde_doc = qi.generate_hypothetical_document(request.query)

    bm25_res = bm25.search(rewritten_query, top_k=5)
    dense_res = dense.search(hyde_doc, top_k=5)

    fused = reciprocal_rank_fusion(bm25_res, dense_res)
    top_candidates = [doc for doc, _ in fused[:3]]

    ranked = reranker.rerank(request.query, top_candidates, top_k=request.top_k)

    formatted_results = []
    for rank, (doc, score) in enumerate(ranked, start=1):
        if score >= 3.0:
            formatted_results.append(
                {
                    "rank": rank,
                    "relevance_score": round(score, 1),
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                }
            )
    return SearchResponse(
        query=request.query,
        rewritten_query=rewritten_query,
        results=formatted_results,
    )


@app.post("/chat", response_model=ChatResponse)
def chat_with_runbooks(request: SearchRequest):

    rewritten_query = qi.rewrite_query(request.query)
    hyde_doc = qi.generate_hypothetical_document(request.query)

    bm25_res = bm25.search(rewritten_query, top_k=5)
    dense_res = dense.search(hyde_doc, top_k=5)

    fused = reciprocal_rank_fusion(bm25_res, dense_res)
    top_candidates = [doc for doc, _ in fused[:3]]

    ranked = reranker.rerank(request.query, top_candidates)
    best_doc, best_score = ranked[0]

    if best_score < 3.0:
        return ChatResponse(
            query=request.query,
            answer="I searched the engineering runbooks, but found no relevant documentation covering your question.",
            source_document_id="NONE",
            source_title="No Relevant Runbook",
            relevance_score=round(best_score, 1),
        )

    generated_answer = generate_grounded_answer(request.query, best_doc)
    return ChatResponse(
        query=request.query,
        answer=generated_answer,
        source_document_id=best_doc["id"],
        source_title=best_doc["title"],
        relevance_score=round(best_score, 1),
    )
