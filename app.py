from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from fastapi import FastAPI
from pydantic import BaseModel  # data validation library
from chainlit.utils import mount_chainlit
from operator import itemgetter

# LLM
llm = ChatOllama(model="qwen3.5:latest", reasoning=False)

# Prompt
prompt_final = ChatPromptTemplate.from_messages(
    [
        ("system", 
            "You are a personal assistant to Josh and Brina.  "
            "You exist to help them answer questions about their lives.  "
            "Use the provided context to help you answer their question: {context}"
        ),
        ("placeholder", "{chat_history}"), # expands a list of message objects e.g. [HumanMessage("what is 1+1"), AIMessage("1+1 = 2")]
        ("human", "{input}"),
    ]
)

# Create retriever from existing Chroma DB
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


# Context Prompting using the Retriever
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "Given the chat history and the latest user question, "
        "rephrase the question as a standalone question that can be understood "
        "without the chat history. Do NOT answer it, just rephrase it."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# History store
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Chain
chain = (
    {
        "context": history_aware_retriever, # context is then populated via create_history_aware_retriever
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")
    } 
    | prompt_final 
    | llm
)
chain_with_history = RunnableWithMessageHistory(
    chain, # pass input and chat_history into chain
    get_session_history, # grabs configurable:session_id from the call, passes it into the session_id param in get_session (runs first)
    input_messages_key="input",
    history_messages_key="chat_history", #inject history from get_session_history into this key, 
)

# Mount UI to fast api app
app = FastAPI()
mount_chainlit(app=app, target="cl_app.py", path="/chainlit")




# Pypandic query class
#class QueryRequest(BaseModel):
#    query: str

# @app.post("/query")
# async def query_model(request: QueryRequest):
#    response = await chain.ainvoke(request.query)
#    return {"answer": response.content}


# async def stream_generator(question: str):
#    async for chunk in chain.astream(question):
#        yield chunk.content


# @app.post("/stream")
# async def stream_query(request: QueryRequest):
#    return StreamingResponse(stream_generator(request.query), media_type="text/plain")
