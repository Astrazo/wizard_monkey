from langchain_ollama import ChatOllama

# LLM
llm = ChatOllama(model="qwen3.5:latest", reasoning=False).with_config(
    run_name="final_llm"
)

classifier_llm = ChatOllama(model="qwen3.5:latest", reasoning=False).with_config(
    run_name="classifier_llm"
)