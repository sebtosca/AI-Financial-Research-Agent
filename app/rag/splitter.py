import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import extract_documents, load_documents

logger = logging.getLogger(__name__)

ENCODING_NAME = os.getenv("TEXT_SPLITTER_ENCODING", "cl100k_base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def validate_splitter_config() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError(
            f"CHUNK_SIZE must be greater than 0. Got: {CHUNK_SIZE}"
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            f"CHUNK_OVERLAP cannot be negative. Got: {CHUNK_OVERLAP}"
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE. "
            f"Got chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}"
        )


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    validate_splitter_config()

    logger.info("Initializing text splitter")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    logger.info(
        (
            "Text splitter configured | "
            "encoding=%s | chunk_size=%d | chunk_overlap=%d | strategy=%s"
        ),
        ENCODING_NAME,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        "recursive_character",
    )

    return text_splitter


def split_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        logger.error("No documents provided for splitting")
        raise ValueError("Cannot split an empty document list")

    text_splitter = build_text_splitter()

    logger.info("Splitting documents | input_documents=%d", len(documents))

    chunks = text_splitter.split_documents(documents)

    logger.info(
        "Documents split successfully | input_documents=%d | output_chunks=%d",
        len(documents),
        len(chunks),
    )

    return chunks


def run_splitter_pipeline() -> list[Document]:
    logger.info("Starting document splitting pipeline")

    extract_documents()
    documents = load_documents()
    chunks = split_documents(documents)

    logger.info(
        "Document splitting pipeline completed | total_chunks=%d",
        len(chunks),
    )

    return chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    chunks = run_splitter_pipeline()