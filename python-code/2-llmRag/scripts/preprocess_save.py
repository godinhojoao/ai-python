# uv run scripts/preprocess_save.py ./scripts/about-joao.pdf ./scripts/vector_store_dir

import os
import pickle
from typing import List

import faiss
import PyPDF2
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_text_from_pdf(pdf_path: str) -> str:
    """Extract and return all text from a PDF file."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def split_text(content: str, chunk_size=500, overlap=50) -> List[str]:
    """Split text into chunks for embedding."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(content)


def create_faiss_vectorstore(chunks: List[str], persist_dir: str):
    """Embed chunks and save a FAISS index to disk."""
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=d) for d in chunks]
    vectordb = FAISS.from_documents(documents, embedding)
    faiss.write_index(vectordb.index, os.path.join(persist_dir, "faiss.index"))
    with open(os.path.join(persist_dir, "docs.pkl"), "wb") as f:
        pickle.dump(vectordb.docstore._dict, f)  # pylint: disable=protected-access
    with open(os.path.join(persist_dir, "index_to_docstore_id.pkl"), "wb") as f:
        pickle.dump(vectordb.index_to_docstore_id, f)
    print(f"FAISS index, docs, and index_to_docstore_id saved to {persist_dir}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal RAG preprocessing")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("persist_dir", help="Directory to save/load FAISS index")
    args = parser.parse_args()

    os.makedirs(args.persist_dir, exist_ok=True)

    index_path = os.path.join(args.persist_dir, "faiss.index")
    docs_path = os.path.join(args.persist_dir, "docs.pkl")
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        print("Extracting and indexing document...")
        PDF_TEXT = load_text_from_pdf(args.pdf_path)
        PDF_CHUNKS = split_text(PDF_TEXT)
        create_faiss_vectorstore(PDF_CHUNKS, args.persist_dir)
    else:
        print("FAISS index already exists in the directory.")
