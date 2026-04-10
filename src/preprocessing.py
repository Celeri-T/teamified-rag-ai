import re
from typing import List

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Rremoves extra whitespace, normalizes newlines, and strips non-printable characters.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    return text.strip()


def preprocess_chunks(chunks: List[Document]) -> List[Document]:
    """Cleans text and filters out chunks that are too small to be useful.

    Args:
        chunks (List[Document]): The split chunks from the PDF.

    Returns:
        List[Document]: List of preprocessed chunks.
    """
    cleaned_chunks = []

    for chunk in chunks:
        # Clean the content
        chunk.page_content = clean_text(chunk.page_content)

        # Skip 2-3 words chunk that lacks context
        if len(chunk.page_content.split()) > 10:
            cleaned_chunks.append(chunk)

    return cleaned_chunks
