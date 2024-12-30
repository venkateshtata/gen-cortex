import sys
from pathlib import Path
import uvicorn
from database.setup import create_document_table
from data_processing.data_extraction import download_docs
from data_processing.data_chunking import chunk_section
from data_processing.embeddings import EmbedChunks
from database.storage import StoreResults
from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
import ray
from data_processing.data_extraction import extract_sections_from_documents
from data_processing.data_extraction import load_documents



def preprocess_and_store_documents():

    # Load documents
    print("Loading documents...")
    ds = load_documents(DATA_DIR)
    print(f"Loaded {ds.count()} documents.")

    # Extract sections from documents
    print("Extracting sections from documents...")
    sections_ds = ds.flat_map(lambda doc: extract_sections_from_documents(doc))


    print(f"Extracted {sections_ds.count()} sections.")

    # Chunk sections
    print("Chunking sections...")
    chunks_ds = sections_ds.flat_map(
    lambda section: chunk_section(section, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
)

    print(f"Generated {chunks_ds.count()} chunks.")

    # Embed chunks
    print("Generating embeddings for chunks...")
    embedder = EmbedChunks(model_name=EMBEDDING_MODEL_NAME)
    embedded_chunks = chunks_ds.map_batches(
        embedder,
        batch_size=100,
        num_gpus=1
    )
    print(f"Generated {embedded_chunks.count()} embeddings.")

    # Store embeddings in the database
    print("Storing embeddings in the database...")
    storage = StoreResults()
    embedded_chunks.map_batches(
        storage,
        batch_size=128,
        num_cpus=1
    ).count()
    print("Successfully stored embeddings in the database.")


def main():
    try:
        sys.path.append(str(Path(__file__).parent.resolve()))

        # Step 1: Create the database table
        print("Setting up the database...")
        create_document_table()

        # Step 2: Download and prepare the documents
        print("Downloading and preparing documents...")
        download_docs(DATA_DIR)

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
