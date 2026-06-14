from unittest.mock import Mock

from app.rag import index as index_module


def test_build_rag_index_uses_existing_pdfs(monkeypatch, tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"pdf")
    documents = [Mock()]
    chunks = [Mock(), Mock()]
    embeddings = Mock()
    extract = Mock()
    build_vectorstore = Mock()

    monkeypatch.setattr(index_module, "DOCS_PATH", tmp_path)
    monkeypatch.setattr(index_module, "CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(index_module, "extract_documents", extract)
    monkeypatch.setattr(index_module, "load_documents", lambda: documents)
    monkeypatch.setattr(index_module, "split_documents", lambda value: chunks)
    monkeypatch.setattr(index_module, "build_embedding_model", lambda: embeddings)
    monkeypatch.setattr(index_module, "build_vectorstore", build_vectorstore)

    index_module.build_rag_index()

    extract.assert_not_called()
    build_vectorstore.assert_called_once_with(
        documents=chunks,
        embedding_model=embeddings,
        collection_name=index_module.CHROMA_COLLECTION_NAME,
        persist_directory=index_module.CHROMA_DB_DIR,
    )


def test_build_rag_index_extracts_when_no_pdfs_exist(monkeypatch, tmp_path):
    extract = Mock()

    monkeypatch.setattr(index_module, "DOCS_PATH", tmp_path)
    monkeypatch.setattr(index_module, "CHROMA_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(index_module, "extract_documents", extract)
    monkeypatch.setattr(index_module, "load_documents", lambda: [Mock()])
    monkeypatch.setattr(index_module, "split_documents", lambda value: [Mock()])
    monkeypatch.setattr(index_module, "build_embedding_model", Mock)
    monkeypatch.setattr(index_module, "build_vectorstore", Mock())

    index_module.build_rag_index()

    extract.assert_called_once_with()


def test_build_rag_index_skips_existing_index(monkeypatch, tmp_path):
    database_path = tmp_path / "chroma"
    database_path.mkdir()
    (database_path / "chroma.sqlite3").touch()
    load_documents = Mock()

    monkeypatch.setattr(index_module, "CHROMA_DB_DIR", str(database_path))
    monkeypatch.setattr(index_module, "RAG_REBUILD_INDEX", False)
    monkeypatch.setattr(index_module, "load_documents", load_documents)

    index_module.build_rag_index()

    load_documents.assert_not_called()
