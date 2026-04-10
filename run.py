from src.config import PDF_FILENAME
from src.document_loader import load_and_chunk_pdf
from src.llm import get_llm
from src.preprocessing import preprocess_chunks
from src.rag_chain import get_rag_chain
from src.utils import clear_console
from src.vector_store import (
    create_vector_store,
    create_vector_store_from_local,
    faiss_index_exists,
)


def main():
    if faiss_index_exists:
        vector_db = create_vector_store_from_local()
    else:
        # Chunk documents
        raw_chunks = load_and_chunk_pdf(PDF_FILENAME)
        processed_chunks = preprocess_chunks(raw_chunks)

        # Initialize FAISS and embed the chunks
        vector_db = create_vector_store()
        vector_db.add_documents(processed_chunks)
    # Clears the console and ask user for query.
    clear_console()

    # Load your local LLM
    llm = get_llm()

    # Create the RAG "Engine"
    rag_engine = get_rag_chain(llm, vector_db)

    query = input("User Query: ")
    results = vector_db.similarity_search(query, k=2)

    print("\nRetrieved Chunks:")
    for chunk in results:
        words = chunk.page_content.split()
        shortened = " ".join(words[:9])

        print(f'- "{shortened}..."')

    response = rag_engine.invoke(query)
    print(f"\nLLM Response:\n{response}")

    # Save FAISS index if not exists
    if not faiss_index_exists():
        vector_db.save_local("faiss_index")


if __name__ == "__main__":
    main()
