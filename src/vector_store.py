from typing import List

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
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


def create_vector_store() -> FAISS:
    """Initialize FAISS using all-MiniLM-L6-v2 as embedding model.

    Returns:
        FAISS: Initialized faiss
    """
    embeddings = get_embedding_model()

    # For FAISS to know vector size using Euclidean distance
    sample_embedding = embeddings.embed_query("hello world")
    index = faiss.IndexFlatL2(len(sample_embedding))

    # vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store
