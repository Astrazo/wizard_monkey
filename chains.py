from prompts import classifer_prompt, llm_prompt
from retriever import history_aware_retriever
from operator import itemgetter
from models import classifier_llm, llm
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda

def log_decision(x):
    print("RAG REQUIRED:", x["decision"])
    return x

# Classifer chain - 
classifer_chain = (
    classifer_prompt | classifier_llm
)
prompt_chain_rag = (
    {
        "context": history_aware_retriever, # context is then populated via create_history_aware_retriever
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")
    } 
    | llm_prompt 
    | llm
)

prompt_chain = (
    {
        "context": lambda _:"No context to add.",
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history")
    } 
    | llm_prompt 
    | llm
)

rag_decision_chain = (
    RunnablePassthrough.assign(
        decision=classifer_chain | (lambda x: x.content.strip().lower()) 
    ) 
    | RunnableLambda(log_decision)
    | RunnableBranch(
        (lambda x: "yes" in x["decision"], prompt_chain_rag),
        prompt_chain
    )
)

