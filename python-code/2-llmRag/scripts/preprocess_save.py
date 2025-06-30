# python3 preprocess_save.py path/to/your.pdf path/to/vectorstore_dir
# python3 scripts/preprocess_save.py ./scripts/about-joao.pdf ./scripts/vector_store_dir

import os
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import faiss
import pickle

def load_text_from_pdf(pdf_path: str) -> str:
    import PyPDF2
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def split_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def create_faiss_vectorstore(docs: List[str], persist_dir: str):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=d) for d in docs]
    vectordb = FAISS.from_documents(documents, embedding)
    faiss.write_index(vectordb.index, os.path.join(persist_dir, "faiss.index"))
    with open(os.path.join(persist_dir, "docs.pkl"), "wb") as f:
        pickle.dump(vectordb.docstore._dict, f)  # save internal dict, not list
    with open(os.path.join(persist_dir, "index_to_docstore_id.pkl"), "wb") as f:
        pickle.dump(vectordb.index_to_docstore_id, f)
    print(f"FAISS index, docs, and index_to_docstore_id saved to {persist_dir}")

def load_faiss_vectorstore(persist_dir):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(persist_dir, "faiss.index"))
    with open(os.path.join(persist_dir, "docs.pkl"), "rb") as f:
        docstore_dict = pickle.load(f)
    with open(os.path.join(persist_dir, "index_to_docstore_id.pkl"), "rb") as f:
        index_to_docstore_id = pickle.load(f)
    vectordb = FAISS(embedding, index, docstore_dict, index_to_docstore_id=index_to_docstore_id)
    return vectordb

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal RAG preprocessing")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("persist_dir", help="Directory to save/load FAISS index")
    args = parser.parse_args()

    os.makedirs(args.persist_dir, exist_ok=True)

    if (
        not os.path.exists(os.path.join(args.persist_dir, "faiss.index"))
        or not os.path.exists(os.path.join(args.persist_dir, "docs.pkl"))
    ):
        print("Extracting and indexing document...")
        text = load_text_from_pdf(args.pdf_path)
        docs = split_text(text)
        create_faiss_vectorstore(docs, args.persist_dir)
    else:
        print("FAISS index already exists in the directory.")
