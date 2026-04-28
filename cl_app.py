import chainlit as cl
from app import chain
from app import chain_with_history
from app import store


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")

    async for chunk in chain_with_history.astream(
        {"input": message.content},
        config={"configurable": {"session_id": cl.user_session.get("id")}},
    ):
        if hasattr(chunk, "content") and chunk.content:
            await msg.stream_token(chunk.content)
    await msg.update()

    print("SESSION ID:", cl.user_session.get("id"))
    print("STORE:", store)
