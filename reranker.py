import re

from dotenv import load_dotenv
from google import genai

load_dotenv()


class Reranker:
    def __init__(self, model_name="gemini-3.5-flash-lite"):
        self.model_name = model_name
        self.client = genai.Client()

    def score_pair(self, query, document):
        prompt = f"""You are an expert retrieval judge evaluating document relevance for engineering runbooks.
Query: {query}
Document Title: {document.get("title", "")}
Document Content:
{document.get("content", "")}
Task: Rate how relevant and actionable this document is for the query on a scale from 0.0 to 10.0.
- 10.0: Directly solves or provides exact instructions for the query.
- 5.0: Related domain or topic, but does not solve the specific problem.
- 0.0: Completely irrelevant.
Respond ONLY with a single number between 0.0 and 10.0. No explanations, no extra words."""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = response.text.strip() if response.text else "0.0"
        match = re.search(r"\d+(\.\d+)?", text)
        if match:
            return float(match.group(0))
        return 0.0

    def rerank(self, query, documents, top_k=None):
        scored_docs = []
        for doc in documents:
            score = self.score_pair(query, doc)
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda item: item[1], reverse=True)

        if top_k is not None:
            return scored_docs[:top_k]
        return scored_docs


if __name__ == "__main__":
    from data import DOCUMENTS

    judge = Reranker()

    query = "PostgreSQL primary crashed, need failover procedure"
    print(f"Query: '{query}'\n")

    # Testing  with 3 documents: one exact match, two partial/irrelevant
    test_docs = [DOCUMENTS[0], DOCUMENTS[1], DOCUMENTS[3]]

    print("Evaluating and scoring candidate documents...")
    ranked = judge.rerank(query, test_docs)

    print("\n=== RERANKED RESULTS ===")
    for doc, score in ranked:
        print(
            f"Relevance Score: {score:.1f} / 10.0 | ID: {doc['id']} | Title: {doc['title']}"
        )
