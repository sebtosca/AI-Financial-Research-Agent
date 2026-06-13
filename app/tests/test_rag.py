import math
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.loader import _safe_extract
from app.rag.retriever import build_retriever
from app.rag.splitter import split_documents
from app.rag.vectorstore import build_vectorstore


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


def test_split_documents_rejects_empty_input():
    with pytest.raises(ValueError, match="Cannot split an empty document list"):
        split_documents([])


def test_safe_extract_rejects_zip_slip(tmp_path: Path):
    zip_path = tmp_path / "unsafe.zip"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("../escape.txt", "should not extract")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        with pytest.raises(RuntimeError, match="Unsafe ZIP path detected"):
            _safe_extract(zip_file, tmp_path / "extract")


def test_vectorstore_rejects_invalid_inputs(tmp_path: Path):
    embeddings = GoldenKeywordEmbeddings()

    with pytest.raises(ValueError, match="No documents provided"):
        build_vectorstore([], embeddings, persist_directory=str(tmp_path))

    with pytest.raises(ValueError, match="Embedding model is missing"):
        build_vectorstore(GOLDEN_DOCUMENTS, None, persist_directory=str(tmp_path))


def test_retriever_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Vector store is missing"):
        build_retriever(None)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        build_retriever(object(), top_k=0)


@pytest.mark.parametrize(("query", "expected_company"), GOLDEN_QUERIES)
def test_golden_dataset_retrieval_returns_expected_document(
    tmp_path: Path,
    query: str,
    expected_company: str,
):
    vectorstore = build_vectorstore(
        documents=GOLDEN_DOCUMENTS,
        embedding_model=GoldenKeywordEmbeddings(),
        collection_name=f"golden_rag_{uuid4().hex}",
        persist_directory=str(tmp_path),
    )

    retriever = build_retriever(vectorstore, top_k=1)
    results = retriever.invoke(query)

    assert results
    assert results[0].metadata["company"] == expected_company
