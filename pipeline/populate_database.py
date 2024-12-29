import os
from utils import *


def populate_database(
    URL,
    DOCS_DIR,
    chunk_size,
    chunk_overlap,
    embed_model,
    embed_batch_size,
    embed_num_gpus,
    embed_compute,
    embed_compute_kwargs,
    db_name,
    user,
    passwd,
    host,
    port,
    store_batch_size,
    store_num_cpus,
    store_compute,
    store_compute_kwargs
    ):
    if os.path.exists(DOCS_DIR):
        print("Using existing documents...")
    else:
        fetch_data(URL=URL, OUTUPUT_DIR=DOCS_DIR)
    
    print("Chunking documents...", end="")
    chunker = DataChunker(
        DOCS_DIR=DOCS_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        URL=URL
    )
    chunks_ds = chunker()
    print("Done")

    print("Embedding chunks...", end="")
    chunks_embeddings = chunks_ds.map_batches(
        ChunkEncoder,
        fn_constructor_kwargs={"model_name": embed_model},
        batch_size=embed_batch_size,
        num_gpus=embed_num_gpus,
        compute=embed_compute(**embed_compute_kwargs)
    )
    print("Done")
    
    print("Populating database...", end="")
    db = DatabaseTool(
        user=user,
        passwd=passwd,
        host=host,
        port=port
    )
    chunks_embeddings.map_batches(
        db.store_embeddings,
        batch_size=store_batch_size,
        num_cpus=store_num_cpus,
        compute=store_compute(**store_compute_kwargs)
    )
    print("Done")
