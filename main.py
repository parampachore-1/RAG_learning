
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
FILE_PATH = Path("data.txt")
DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_documents"


def main():
    # 1. Load document
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH.resolve()}")

    print("Loading document...")
    documents = TextLoader(
        str(FILE_PATH),
        encoding="utf-8"
    ).load()

    # 2. Split document
    print("Splitting document...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks found in the document.")

    print(f"Total chunks: {len(chunks)}")

    # 3. Load local embedding model
    print("Loading Sentence Transformers model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Create/open persistent vector database
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 5. Generate embeddings and store chunks
    print("Generating embeddings and storing in ChromaDB...")

    ids = [
        f"data-txt-chunk-{i}"
        for i in range(len(chunks))
    ]

    vector_db.add_documents(
        documents=chunks,
        ids=ids
    )

    # 6. Verify database
    print("\nIngestion completed successfully!")
    print("Chunks processed:", len(chunks))
    print("Vectors stored:", vector_db._collection.count())
    print("Database location:", Path(DB_PATH).resolve())

    # 7. Test semantic search
    results = vector_db.similarity_search(
        "What information is available in this document?",
        k=min(3, len(chunks))
    )

    print("\nSample retrieved chunks:")
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(doc.page_content)
        print("Metadata:", doc.metadata)


if __name__ == "__main__":
    main()
