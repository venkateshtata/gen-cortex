import ray
from pathlib import Path
import os
from data_utils import *
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from ray.data import ActorPoolStrategy
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial


EFS_DIR = os.environ["EFS_DIR"]
DOCS_DIR = Path(EFS_DIR, "docs.ray.io/en/master/")

ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])

print(f"{ds.count()} documents")

sections_ds = ds.flat_map(extract_sections)
sections = sections_ds.take_all()

print (len(sections))


# Splitting Text
chunk_size = 300
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)



def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.create_documents(
        texts=[section["text"]], 
        metadatas=[{"source": section["source"]}])
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]



# Scale chunking
chunks_ds = sections_ds.flat_map(partial(
    chunk_section, 
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap))

print(f"{chunks_ds.count()} chunks")

chunks_ds.show(1)
print("That was the first processed chunk\n\n\n")



# Get Embeddings
class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 512}
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"], 
            "source": batch["source"],
            "embeddings": embeddings
        }
        

embedding_model_name = "thenlper/gte-base"

embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=128,  # Reduce from 512
    concurrency=1,   # Match the single GPU
    num_gpus=1,
    compute=ActorPoolStrategy(size=1),
)


# Fix the device check - create a temporary instance to check the device
embed_chunks = EmbedChunks(embedding_model_name)
print(f"Model running on device: {embed_chunks.embedding_model.client.device}")

# Add progress tracking
total = chunks_ds.count()
processed = embedded_chunks.count()
print(f"Processed {processed}/{total} chunks")
print("Done!")