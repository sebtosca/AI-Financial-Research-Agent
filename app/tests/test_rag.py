import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from app.eval.fixtures import GOLDEN_DOCUMENTS, GOLDEN_QUERIES, GoldenKeywordEmbeddings
from app.rag.loader import _safe_extract
from app.rag.retriever import build_retriever
from app.rag.splitter import split_documents
from app.rag.vectorstore import build_vectorstore


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
