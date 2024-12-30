import os
import ray
import sys
import warnings 
from dotenv import load_dotenv
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from ray.data import ActorPoolStrategy
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.data import extract_sections
from functools import partial
import subprocess



sys.path.append("..")
warnings.filterwarnings("ignore")
load_dotenv()

# Add wget command to download docs
data_dir = Path('./data')

if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
    print('Data exists already!')
else:
    data_dir.mkdir(exist_ok=True)  # Create data directory if it doesn't exist
    wget_command = [
        'wget',
        '-e', 'robots=off',
        '--recursive',
        '--no-clobber',
        '--page-requisites',
        '--html-extension',
        '--convert-links',
        '--restrict-file-names=windows',
        '--domains', 'docs.ray.io',
        '--no-parent',
        '--accept=html',
        '-P', str(data_dir),
        'https://docs.ray.io/en/master/'
    ]

    # Execute wget command
    subprocess.run(wget_command)


DOCS_DIR = Path('./data', "docs.ray.io/en/master/")
ds = ray.data.from_items([{"path": str(path)} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])
print(f"{ds.count()} documents")


# Extract sections
sections_ds = ds.flat_map(extract_sections)
sections = sections_ds.take_all()
print ('Total Sections: ', len(sections))


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
    
    # Extract text and source from the section
    section_text = section["text"]
    section_source = section["source"]
    
    chunks = text_splitter.create_documents(
        texts=[section_text], 
        metadatas=[{"source": section_source}]
    )
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]


# Scale chunking
chunks_ds = sections_ds.flat_map(partial(
    chunk_section, 
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap))

print(f"{chunks_ds.count()} chunks")
chunks_ds.show(1)



class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100})

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}


# Embed chunks
embedding_model_name = "thenlper/gte-base"
embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=100, 
    num_gpus=1,
    compute=ActorPoolStrategy(size=1))

print('Total embeddings: ', embedded_chunks.count())
print('Embedding length: ', len(embedded_chunks.take(1)[0]['embeddings']))
