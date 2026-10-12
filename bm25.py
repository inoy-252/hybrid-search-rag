import math
import re
from collections import Counter


def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9_\-\/]+", text.lower())
    return tokens


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus_size = len(documents)

        self.doc_tokens = []
        self.doc_lens = []
        for doc in documents:
            doc_cont_titl = doc["content"] + " " + doc["title"]
            tokens = tokenize(doc_cont_titl)

            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))

        self.avg_doc_length = sum(self.doc_lens) / (self.corpus_size)

        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        self.doc_freqs = Counter()
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def score_document(self, query_tokens, doc_index):
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        term_freqs = self.doc_term_freqs[doc_index]
        for token in query_tokens:
            if token not in term_freqs:
                continue
            tf = term_freqs[token]
            idf = self.idf.get(token, 0.0)

            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * (doc_len / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)
        return score

    def search(self, query, top_k=None):
        query_words = tokenize(query)
        results = []
        for i, doc in enumerate(self.documents):
            score = self.score_document(query_words, i)
            results.append((doc, score))

        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            return results[:top_k]
        return results


if __name__ == "__main__":
    from data import DOCUMENTS

    engine = BM25(DOCUMENTS)

    test_query = "ERR-PAY-502-GATEWAY"
    results = engine.search(test_query, top_k=3)

    print(f"Query: '{test_query}'\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | ID: {doc['id']} | Title: {doc['title']}")
