import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()


def cosine_similarity(vec_1, vec_2):
    norm1 = np.linalg.norm(vec_1)
    norm2 = np.linalg.norm(vec_2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec_1, vec_2) / (norm1 * norm2))


class DenseRetriever:
    def __init__(self, documents, model_name="gemini-embedding-2"):
        self.documents = documents
        self.model_name = model_name
        self.client = genai.Client()

        print("Generating embeddings for documents...")
        self.doc_embeddings = []
        for doc in self.documents:
            text = f"{doc['title']}\n{doc['content']}"
            embedding = self.get_embedding(text)
            self.doc_embeddings.append(embedding)
        self.doc_embeddings = np.array(self.doc_embeddings)

    def get_embedding(self, text):
        response = self.client.models.embed_content(
            model=self.model_name, contents=text
        )
        if not response.embeddings:
            raise ValueError(f"No embeddings returned for: {text[:30]}...")
        return np.array(response.embeddings[0].values)

    def search(self, query, top_k=None):
        query_vec = self.get_embedding(query)
        results = []
        for i, doc in enumerate(self.documents):
            doc_vec = self.doc_embeddings[i]
            score = cosine_similarity(query_vec, doc_vec)
            results.append((doc, score))

        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            return results[:top_k]
        return results


if __name__ == "__main__":
    from data import DOCUMENTS

    retriever = DenseRetriever(DOCUMENTS)
    test_query = "users cannot sign in due to expired credentials"
    results = retriever.search(test_query, top_k=3)
    print(f"\nQuery: '{test_query}'\n")
    for doc, score in results:
        print(
            f"Cosine Similarity: {score:.4f} | ID: {doc['id']} | Title: {doc['title']}"
        )
