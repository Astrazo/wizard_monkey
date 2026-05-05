from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from models import llm
from prompts import rag_prompt

embed_function = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    collection_name="home_llm",
    embedding_function=embed_function,
    persist_directory="chroma_db",
)
retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, rag_prompt
)
