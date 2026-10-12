import time

from bm25 import BM25
from data import DOCUMENTS
from dense_retriever import DenseRetriever
from fusion import reciprocal_rank_fusion
from query_intelligence import QueryIntelligence
from reranker import Reranker

EVAL_DATASET = [
    {
        "query": "how to stop getting charged twice if a user double clicks pay",
        "expected_id": "DOC-PAY-404",
        "type": "Conversational / Vague",
    },
    {
        "query": "ERR-AUTH-401-EXPIRED clock drift",
        "expected_id": "DOC-AUTH-101",
        "type": "Exact Error Code",
    },
    {
        "query": "primary database is down how to promote replica with patronictl",
        "expected_id": "DOC-DB-202",
        "type": "Technical Procedure",
    },
    {
        "query": "my website is overwhelmed with traffic what should I do",
        "expected_id": "DOC-INFRA-303",
        "type": "Conversational / Vague",
    },
    {
        "query": "how to dump heap metrics for memory leaks with SIGUSR1",
        "expected_id": "DOC-OBS-505",
        "type": "Technical Debugging",
    },
    {
        "query": "what is the timeout retry policy when webhook callbacks fail",
        "expected_id": "DOC-PAY-404",
        "type": "Specific Specification",
    },
    {
        "query": "port 6432 connection pooler pgbouncer setup",
        "expected_id": "DOC-DB-202",
        "type": "Port / Configuration",
    },
    {
        "query": "how to revoke all user sessions when credentials are leaked",
        "expected_id": "DOC-AUTH-101",
        "type": "Security Incident",
    },
]

print("Initializing search engines for evaluation...")
bm25 = BM25(DOCUMENTS)
dense = DenseRetriever(DOCUMENTS)
qi = QueryIntelligence()
reranker = Reranker()

print("Starting evaluation loop...\n")

hits_at_1 = 0
reciprocal_ranks = []

for i, test_case in enumerate(EVAL_DATASET, start=1):
    query = test_case["query"]
    expected_id = test_case["expected_id"]

    rewritten_query = qi.rewrite_query(query)
    hyde_doc = qi.generate_hypothetical_document(query)

    bm25_res = bm25.search(rewritten_query, top_k=5)
    dense_res = dense.search(hyde_doc, top_k=5)

    fused = reciprocal_rank_fusion(bm25_res, dense_res)
    top_candidates = [doc for doc, _ in fused[:3]]
    ranked_results = reranker.rerank(query, top_candidates)

    found_rank = None
    for rank, (doc, score) in enumerate(ranked_results, start=1):
        if doc["id"] == expected_id:
            found_rank = rank
            break

    if found_rank == 1:
        status = "PASS(RANK #1)"
        hits_at_1 += 1
        reciprocal_ranks.append(1.0)
    elif found_rank is not None:
        status = f"PARTIAL (RANK #{found_rank})"
        reciprocal_ranks.append(1.0 / found_rank)

    else:
        status = "FAIL (NOT IN TOP 3)"
        reciprocal_ranks.append(0.0)

    print(f"[{i}/{len(EVAL_DATASET)}] {status} -> Expected: {expected_id}")
    print(f'       Question: "{query}"\n')

    print("Pacing API requests (sleeping 4s)...")
    time.sleep(4)


total_queries = len(EVAL_DATASET)
hit_rate_1 = (hits_at_1 / total_queries) * 100
mrr = sum(reciprocal_ranks) / total_queries
print("=" * 60)
print(" BENCHMARK EVALUATION REPORT")
print("=" * 60)
print(f"Total Test Cases           : {total_queries}")
print(f"Hit Rate @ 1 (Top-1 Acc)   : {hit_rate_1:.1f}% ({hits_at_1}/{total_queries})")
print(f"MRR (Mean Reciprocal Rank) : {mrr:.3f} / 1.000")
print("=" * 60)
