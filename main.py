import sys
from pathlib import Path
import uvicorn
from database.setup import create_document_table
from data_processing.data_extraction import download_docs
from data_processing.data_chunking import chunk_section
from data_processing.embeddings import EmbedChunks
from database.storage import StoreResults
from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from data_processing.data_extraction import extract_sections_from_documents
from data_processing.data_extraction import load_documents
from time import time



def preprocess_and_store_documents():
    start_total = time()

    # Load documents
    print("Loading documents...")
    start = time()
    ds = load_documents(DATA_DIR)
    print(f"Loaded {ds.count()} documents in {time() - start:.2f} seconds.")

    # Extract sections from documents
    print("Extracting sections from documents...")
    start = time()
    sections_ds = ds.flat_map(lambda doc: extract_sections_from_documents(doc))
    print(f"Extracted {sections_ds.count()} sections in {time() - start:.2f} seconds.")

    # Chunk sections
    print("Chunking sections...")
    start = time()
    chunks_ds = sections_ds.flat_map(
        lambda section: chunk_section(section, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )
    print(f"Generated {chunks_ds.count()} chunks in {time() - start:.2f} seconds.")

    # Embed chunks
    print("Generating embeddings for chunks...")
    start = time()
    embedder = EmbedChunks(model_name=EMBEDDING_MODEL_NAME)
    embedded_chunks = chunks_ds.map_batches(
        embedder,
        batch_size=100,
        num_gpus=1
    )
    print(f"Generated {embedded_chunks.count()} embeddings in {time() - start:.2f} seconds.")

    # Store embeddings in the database
    print("Storing embeddings in the database...")
    start = time()
    storage = StoreResults()
    embedded_chunks.map_batches(
        storage,
        batch_size=128,
        num_cpus=1
    ).count()
    print(f"Successfully stored embeddings in the database in {time() - start:.2f} seconds.")
    
    print(f"Total processing time: {time() - start_total:.2f} seconds")


def main():
    try:
        start_total = time()
        sys.path.append(str(Path(__file__).parent.resolve()))

        # Step 1: Create the database table
        print("Setting up the database...")
        start = time()
        create_document_table()
        print(f"Database setup completed in {time() - start:.2f} seconds")

        # Step 2: Download and prepare the documents
        print("Downloading and preparing documents...")
        start = time()
        download_docs(DATA_DIR)
        print(f"Document download completed in {time() - start:.2f} seconds")

        # Step 3: Process documents and store embeddings
        print("Processing documents and storing embeddings...")
        preprocess_and_store_documents()

        # Step 4: Start the FastAPI application
        print("Starting the FastAPI server...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
