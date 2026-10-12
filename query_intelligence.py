import time

from dotenv import load_dotenv
from google import genai


class QueryIntelligence:
    def __init__(self, model_name="gemini-3.5-flash-lite"):
        load_dotenv()
        self.client = genai.Client()
        self.model_name = model_name

    def _generate_with_retry(self, prompt: str, max_retries: int = 5) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(" Rate limit reached (429). Pausing 25s before retry...")
                    time.sleep(25)
                else:
                    raise
        return ""

    def rewrite_query(self, query: str) -> str:
        prompt = f"""You are a technical search query rewriting engine for an internal engineering knowledge base.
The user's question may be informal, conversational, or vague.

Your job:
1. Identify the core technical intent behind the user's question.
2. Expand it with relevant technical keywords, domain terminology, error codes, or infrastructure concepts that would appear in formal engineering runbooks.
3. Return ONLY the rewritten keyword query. Do NOT add preamble, quotation marks, or explanations.

User Query: {query}
Rewritten Technical Search Query:"""

        text = self._generate_with_retry(prompt)
        return text if text else query

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = f"""You are a senior systems infrastructure engineer writing internal runbooks.
Given the engineer's question below, write a short, highly technical 2-to-3 sentence documentation excerpt that directly answers it.
Use authoritative technical terminology, architectural mechanisms, and typical error codes.
Return ONLY the raw documentation paragraph. Do NOT add greetings, titles, or explanations.

Engineer's Question: {query}
Hypothetical Documentation Excerpt:"""

        text = self._generate_with_retry(prompt)
        return text if text else query


if __name__ == "__main__":
    qi = QueryIntelligence()
    test_q = "how to stop getting charged twice if a user double clicks pay"

    print("\n" + "=" * 65)
    print(f"Original Query:\n{test_q}\n")

    rewritten = qi.rewrite_query(test_q)
    print(f"1. Rewritten Query (Keywords for BM25):\n{rewritten}\n")

    hypo_doc = qi.generate_hypothetical_document(test_q)
    print(f"2. HyDE Document (Full Paragraph for Dense Vectors):\n{hypo_doc}")
    print("=" * 65)
