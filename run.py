from src.config import PDF_FILENAME
from src.document_loader import load_and_chunk_pdf
from src.llm import get_llm
from src.utils import clear_console
from src.vector_store import create_vector_store


def main():
    # Chunk documents and store in FAISS vector store
    chunks = load_and_chunk_pdf(PDF_FILENAME)
    vector_db = create_vector_store(chunks)
    vector_db.add_documents(chunks)

    # Clears the console and ask user for query.
    # clear_console()
    query = input("User Query: ")
    results = vector_db.similarity_search(query, k=2)

    print("\nRetrieved Chunks:")
    for chunk in results:
        words = chunk.page_content.split()
        shortened = " ".join(words[:9])

        print(f'- "{shortened}..."')


if __name__ == "__main__":
    main()
