import sys
from pathlib import Path
import uvicorn
from database.setup import create_document_table
from data_processing.data_extraction import download_docs
from config.settings import DATA_DIR

def main():
    try:
        # Add `rag_pipeline` to the Python path
        sys.path.append(str(Path(__file__).parent.resolve()))

        # Step 1: Create the database table
        print("Setting up the database...")
        create_document_table()

        # Step 2: Download and prepare the documents
        print("Downloading and preparing documents...")
        download_docs(DATA_DIR)

        # Step 3: Start the FastAPI application
        print("Starting the FastAPI server...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
