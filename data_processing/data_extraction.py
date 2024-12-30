from pathlib import Path
from rag.data import extract_sections
from config.settings import DATA_DIR, DOCS_URL
import os
import subprocess

def download_docs(data_dir=DATA_DIR, docs_url=DOCS_URL):
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print('Data exists already!')
        return
    Path(data_dir).mkdir(exist_ok=True)
    wget_command = [
        'wget', '-e', 'robots=off', '--recursive', '--no-clobber',
        '--page-requisites', '--html-extension', '--convert-links',
        '--restrict-file-names=windows', '--domains', 'docs.ray.io',
        '--no-parent', '--accept=html', '-P', data_dir, docs_url
    ]
    subprocess.run(wget_command)

def extract_sections_from_documents(document):
    """Extract sections from the document."""
    # Ensure the function receives a dictionary and passes it correctly
    return extract_sections(document)



import ray

def load_documents(data_dir):
    """Load document contents into a Ray Dataset."""
    documents = []
    for path in Path(data_dir).rglob("*.html"):
        if not path.is_dir():
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append({"text": content, "source": str(path)})
    return ray.data.from_items(documents)
