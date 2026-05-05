from langchain_core.prompts import ChatPromptTemplate

# Classifer prompt - Does this need RAG
classifer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are a routing classifier. "
        "Answer only 'yes' or 'no'. "
        "Should we look up personal information about Josh and Brina to answer this question? "
        "Answer 'no' only if the question is clearly general knowledge: maths, science, coding, geography, history, or world facts. "
        "If there is any chance it relates to Josh and Brina personally, answer 'yes'. "
        "False negatives are very costly to the operation.  Only return no if you are very sure of yourself. "
    ),
    ("human", "{input}")
])

# Rag Prompt - rephrase the question for better RAG lookup
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "Given the chat history and the latest user question, "
        "rephrase the question as a standalone question that can be understood "
        "without the chat history. Do NOT answer it, just rephrase it."
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

# LLM Prompt - what is fed to the model
llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            "You are a personal assistant to Josh and Brina.  You should ask who you are speaking to, if you don't know."
            "You exist to help them answer questions about their lives.  "
            "Use the provided context to help you answer their question: {context}"
        ),
        ("placeholder", "{chat_history}"), # expands a list of message objects e.g. [HumanMessage("what is 1+1"), AIMessage("1+1 = 2")]
        ("human", "{input}"),
    ]
)