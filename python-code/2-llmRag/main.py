"""Entry point for the RAG chatbot example."""

from llm_utils import get_chat_completion_rag, get_chat_completion

VECTORSTORE_DIR = "./scripts/vector_store_dir"
response_rag = get_chat_completion_rag("Who is João Godinho?", VECTORSTORE_DIR)
response = get_chat_completion("Who is João Godinho?")

print(f"\n\nCOMMON RESPONSE: {response}\n\n")
print(f"RESPONSE RAG: {response_rag}")
