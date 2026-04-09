from src.config import PDF_FILENAME
from src.document_loader import load_and_chunk_pdf
from src.utils import clear_console
from src.vector_store import create_vector_store


def main():
    # 1. Ingestion
    chunks = load_and_chunk_pdf(PDF_FILENAME)

    # 2. Vectorization
    vector_db = create_vector_store(chunks)

    clear_console()

    # Quick Test: Search the vector store
    query = input("User Query: ")
    results = vector_db.similarity_search(query, k=2)

    print("\nRetrieved Chunks:")
    for chunk in results:
        words = chunk.page_content.split()
        shortened = " ".join(words[:9])

        print(f'- "{shortened}..."')


if __name__ == "__main__":
    main()
