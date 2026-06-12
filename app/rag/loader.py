import logging
import os
import zipfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DOCS_PATH = Path(
    os.getenv(
        "DOCS_PATH",
        "/home/professorx/projects/Agentic_RAG/app/docs",
    )
)

ZIP_FILE = Path(
    os.getenv(
        "ZIP_FILE",
        DOCS_PATH / "Companies-AI-Initiatives.zip",
    )
)


def _safe_extract(zip_ref: zipfile.ZipFile, extract_to: Path) -> None:
    extract_to = extract_to.resolve()

    for member in zip_ref.namelist():
        target_path = (extract_to / member).resolve()

        if not str(target_path).startswith(str(extract_to)):
            raise RuntimeError(
                f"Unsafe ZIP path detected: {member}"
            )

    zip_ref.extractall(extract_to)


def extract_documents(force: bool = False) -> None:
    logger.info("Starting document extraction process")

    if not ZIP_FILE.exists():
        logger.error("ZIP file not found: %s", ZIP_FILE)
        raise FileNotFoundError(
            f"ZIP file does not exist: {ZIP_FILE}"
        )

    DOCS_PATH.mkdir(parents=True, exist_ok=True)

    existing_pdfs = list(DOCS_PATH.rglob("*.pdf"))

    if existing_pdfs and not force:
        logger.info(
            "Skipping extraction | existing_pdfs=%d | docs_path=%s",
            len(existing_pdfs),
            DOCS_PATH,
        )
        return

    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            logger.info(
                "Extracting ZIP file | zip_file=%s | target_dir=%s",
                ZIP_FILE,
                DOCS_PATH,
            )

            _safe_extract(zip_ref, DOCS_PATH)

            logger.info(
                "Successfully extracted ZIP file | target_dir=%s",
                DOCS_PATH,
            )

    except zipfile.BadZipFile as e:
        logger.exception("Invalid or corrupted ZIP file")
        raise zipfile.BadZipFile(
            f"Failed to extract corrupted ZIP file: {ZIP_FILE}"
        ) from e

    except Exception as e:
        logger.exception("Unexpected extraction error")
        raise RuntimeError(
            f"Unexpected error during ZIP extraction: {e}"
        ) from e


def load_documents() -> list[Document]:
    logger.info("Loading PDF documents")

    pdf_files = list(DOCS_PATH.rglob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found | docs_path=%s", DOCS_PATH)
        raise ValueError(
            f"No PDF files found inside: {DOCS_PATH}"
        )

    logger.info(
        "PDF files found | count=%d | docs_path=%s",
        len(pdf_files),
        DOCS_PATH,
    )

    try:
        loader = PyPDFDirectoryLoader(
            path=str(DOCS_PATH),
            recursive=True,
        )

        documents = loader.load()

        logger.info(
            "Successfully loaded PDF pages | pages=%d",
            len(documents),
        )

        return documents

    except Exception as e:
        logger.exception("Failed to load PDF documents")
        raise RuntimeError(
            f"Document loading failed: {e}"
        ) from e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    extract_documents()
    documents = load_documents()

    logger.info(
        "Loader pipeline completed successfully | documents=%d",
        len(documents),
    )