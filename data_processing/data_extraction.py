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

def extract_sections_from_documents(ds):
    """Extract sections from the document dataset."""
    sections_ds = ds.flat_map(extract_sections)
    return sections_ds
