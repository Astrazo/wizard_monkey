from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from chains import rag_decision_chain


store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_decision_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
