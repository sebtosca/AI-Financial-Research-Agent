"""Shared golden RAG fixtures used by both the deterministic test suite
(app/tests/test_rag.py) and the evaluation harness (app/eval/).
"""

import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class GoldenKeywordEmbeddings(Embeddings):
    """Deterministic embeddings for RAG tests without external APIs."""

    vocabulary = [
        "amazon",
        "aws",
        "bedrock",
        "blackwell",
        "copilot",
        "gpu",
        "ibm",
        "microsoft",
        "nvidia",
        "watsonx",
    ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        vector = [float(text.count(term)) for term in self.vocabulary]
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]


GOLDEN_DOCUMENTS = [
    Document(
        page_content=(
            "Microsoft expands Azure AI and Copilot capabilities for "
            "enterprise productivity workflows."
        ),
        metadata={"company": "MSFT", "topic": "azure-copilot"},
    ),
    Document(
        page_content=(
            "NVIDIA Blackwell GPU platforms accelerate model training "
            "and inference for AI infrastructure."
        ),
        metadata={"company": "NVDA", "topic": "blackwell-gpu"},
    ),
    Document(
        page_content=(
            "Amazon AWS Bedrock gives teams managed access to foundation "
            "models for generative AI applications."
        ),
        metadata={"company": "AMZN", "topic": "aws-bedrock"},
    ),
    Document(
        page_content=(
            "IBM watsonx supports enterprise AI governance, model lifecycle "
            "management, and trusted deployment."
        ),
        metadata={"company": "IBM", "topic": "watsonx-governance"},
    ),
]

GOLDEN_QUERIES = [
    ("Which company discusses Azure Copilot?", "MSFT"),
    ("Who makes Blackwell GPUs for AI infrastructure?", "NVDA"),
    ("Which document is about AWS Bedrock foundation models?", "AMZN"),
    ("Which company focuses on watsonx AI governance?", "IBM"),
]
