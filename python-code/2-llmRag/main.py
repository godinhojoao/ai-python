from llm_utils import getChatCompletionRag, getChatCompletion

vectorstore_dir = './scripts/vector_store_dir'
responseRag = getChatCompletionRag("Who is João Godinho?", vectorstore_dir)
response = getChatCompletion("Who is João Godinho?")

print(f"\n\nCOMMON RESPONSE: {response}\n\n")

print(f"RESPONSE RAG: {responseRag}")
