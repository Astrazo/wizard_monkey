from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel  # data validation library
from chainlit.utils import mount_chainlit


# Load existing Chroma DB
embed_function = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    collection_name="home_llm",
    embedding_function=embed_function,
    persist_directory="chroma_db",
)

# Create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Use the context below to answer the question.
If you don't know, just say you don't know.

Context: {context}

Question: {question}
"""
)

# LLM
llm = ChatOllama(model="qwen3.5:latest", reasoning=False)


# Pypandic query class
class QueryRequest(BaseModel):
    query: str


# Ask
app = FastAPI()

# Chain
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm


@app.post("/query")
async def query_model(request: QueryRequest):
    response = await chain.ainvoke(request.query)
    return {"answer": response.content}


async def stream_generator(question: str):
    async for chunk in chain.astream(question):
        yield chunk.content


@app.post("/stream")
async def stream_query(request: QueryRequest):
    return StreamingResponse(stream_generator(request.query), media_type="text/plain")


mount_chainlit(app=app, target="cl_app.py", path="/chainlit")
