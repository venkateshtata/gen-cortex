import sys
from pathlib import Path
from time import time, sleep
from database.setup import create_document_table
from data_processing.data_extraction import download_docs, load_documents, extract_sections_from_documents
from data_processing.data_chunking import chunk_section
from data_processing.embeddings import EmbedChunks
from database.storage import StoreResults
from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
import ray
from ray import serve
from app import QueryService, app 
import uvicorn
from app import app

ray.init(ignore_reinit_error=True)
serve.start()

def preprocess_and_store_documents():
    start_total = time()

    print("Loading documents...")
    start = time()
    ds = load_documents(DATA_DIR)
    print(f"Loaded {ds.count()} documents in {time() - start:.2f} seconds.")

    print("Extracting sections from documents...")
    start = time()
    sections_ds = ds.flat_map(lambda doc: extract_sections_from_documents(doc))
    print(f"Extracted {sections_ds.count()} sections in {time() - start:.2f} seconds.")

    print("Chunking sections...")
    start = time()
    chunks_ds = sections_ds.flat_map(
        lambda section: chunk_section(section, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )
    print(f"Generated {chunks_ds.count()} chunks in {time() - start:.2f} seconds.")

    print("Generating embeddings for chunks...")
    start = time()
    embedder = EmbedChunks(model_name=EMBEDDING_MODEL_NAME)
    embedded_chunks = chunks_ds.map_batches(
        embedder, batch_size=100, num_gpus=1
    )
    print(f"Generated {embedded_chunks.count()} embeddings in {time() - start:.2f} seconds.")

    print("Storing embeddings in the database...")
    start = time()
    storage = StoreResults()
    embedded_chunks.map_batches(
        storage, batch_size=128, num_cpus=1
    ).count()
    print(f"Successfully stored embeddings in the database in {time() - start:.2f} seconds.")

    print(f"Total processing time: {time() - start_total:.2f} seconds")



def main():
    try:
        sys.path.append(str(Path(__file__).parent.resolve()))

        ray.init(ignore_reinit_error=True)
        serve.start()

        print("Setting up the database...")
        create_document_table()

        print("Downloading and preparing documents...")
        download_docs(DATA_DIR)

        # preprocess_and_store_documents()

        print("Deploying the QueryService with Ray Serve...")
        serve.run(QueryService.bind())  # Correct way to deploy with Ray Serve


        # print("Starting the FastAPI server with Ray Serve...")
        # uvicorn.run(app, host="0.0.0.0", port=8000)

        while True:
            sleep(60)  # Keep the process alive indefinitely

        # print("Starting the FastAPI server with Ray Serve...")
        # serve.run(app)  # Bind the app to Ray Serve
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()