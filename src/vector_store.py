from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initializes the HuggingFace embedding model defined in config.

    Returns:
        HuggingFaceEmbeddings: The model imported from HuggingFace.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # To make it run on CPU
        model_kwargs={"device": "cpu"},
    )


def create_vector_store(chunks: List[Document]) -> FAISS:
    """Takes text chunks, embeds them, and stores them in a FAISS index.

    Args:
        chunks (List[Document]): The list containing the chunks from the text splitter.

    Returns:
        FAISS: The initialized FAISS from documents and embeddings.
    """
    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save locally so you don't have to re-embed every time you run
    vector_store.save_local(FAISS_INDEX_PATH)

    return vector_store
