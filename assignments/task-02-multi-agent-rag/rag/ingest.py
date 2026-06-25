import os
import logging
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "documents"))
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))

def ingest_docs() -> None:
    """Load, chunk, embed, and store documents in local ChromaDB idempotently."""
    logger.info("Starting document ingestion process...")

    if not os.path.exists(DOCS_DIR):
        logger.error(f"Documents directory '{DOCS_DIR}' not found. Cannot proceed.")
        return

    # 1. Initialize embeddings
    logger.info("Initializing HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Initialize ChromaDB
    logger.info(f"Connecting to ChromaDB at '{DB_DIR}'...")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    # 3. Read files and split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Track files processed
    files_processed = 0
    total_chunks_added = 0

    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith((".txt", ".md")):
                file_path = os.path.join(root, file)
                logger.info(f"Processing document: {file_path}")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    doc = Document(page_content=content, metadata={"source": file})
                    chunks = text_splitter.split_documents([doc])
                    
                    if not chunks:
                        logger.warning(f"File {file} is empty. Skipping.")
                        continue

                    # 4. Enforce idempotency: delete previous vectors for this file source
                    # Generating deterministic IDs: filename_chunkIndex
                    ids = [f"{file}_{idx}" for idx in range(len(chunks))]
                    
                    try:
                        # Clean up existing documents with this source metadata to prevent orphans
                        # if the document length changed.
                        db.delete(where={"source": file})
                        logger.info(f"Cleared pre-existing vectors for source '{file}'")
                    except Exception as clean_err:
                        logger.debug(f"Clear failed (may be an empty collection): {clean_err}")

                    # 5. Add documents
                    db.add_documents(chunks, ids=ids)
                    logger.info(f"Ingested {len(chunks)} chunks for '{file}' with IDs {ids[0]} to {ids[-1]}")
                    
                    files_processed += 1
                    total_chunks_added += len(chunks)

                except Exception as e:
                    logger.error(f"Failed to ingest file '{file_path}': {e}", exc_info=True)

    logger.info(f"Ingestion complete. Processed {files_processed} files and added {total_chunks_added} chunks.")

if __name__ == "__main__":
    ingest_docs()
