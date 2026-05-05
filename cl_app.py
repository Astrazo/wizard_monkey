import chainlit as cl
from app import chain_with_history, store


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")

    async for event in chain_with_history.astream_events(
        {"input": message.content},
        config={"configurable": {"session_id": cl.user_session.get("id")}},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream" and event["name"] == "final_llm":
            chunk = event["data"]["chunk"]
            await msg.stream_token(chunk.content)
    await msg.update()
