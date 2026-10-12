def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    rrf_scores = {}
    doc_lookup = {}
    for result_list in [bm25_results, dense_results]:
        for rank, (doc, _) in enumerate(result_list, start=1):
            doc_id = doc["id"]
            doc_lookup[doc_id] = doc

            points = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + points

    sorted_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    fused_results = []
    for doc_id, score in sorted_scores:
        doc = doc_lookup[doc_id]
        fused_results.append((doc, score))

    return fused_results


if __name__ == "__main__":
    from bm25 import BM25
    from data import DOCUMENTS
    from dense_retriever import DenseRetriever

    print("Initializing BM25 and Dense Retriever...")
    bm25 = BM25(DOCUMENTS)
    dense = DenseRetriever(DOCUMENTS)

    query = "reconciliation timeout ERR-PAY-502-GATEWAY"
    print(f"\nSearching for: '{query}'\n")

    bm25_res = bm25.search(query, top_k=3)
    dense_res = dense.search(query, top_k=3)

    fused = reciprocal_rank_fusion(bm25_res, dense_res)

    print("=== HYBRID SEARCH RESULTS (RRF) ===")
    for doc, rrf_score in fused[:3]:
        print(f"RRF Score: {rrf_score:.5f} | ID: {doc['id']} | Title: {doc['title']}")
