from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_and_chunk_pdf(file_name: str) -> list[Document]:
    """Loads a PDF using PyMuPDF for text extraction and splits it into chunks.

    Args:
        file_name (str): File name of the PDF to be loaded and chunked.

    Raises:
        FileNotFoundError: If the test PDF is not found in data directory.

    Returns:
        list[Document]: A list containing the resulting chunks.
    """
    file_path = Path(file_name)

    if not file_path.exists():
        raise FileNotFoundError(f"Error: Could not find {file_path}.")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    return chunks
