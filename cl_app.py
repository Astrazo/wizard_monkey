import chainlit as cl
from app import chain


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")

    # chain.astream(message.content) starts the pipeline
    # async for chunk in... means whenever it returns a chunk, do something
    # msg.stream_token (appends the token to the UI in real time)
    # msg.update (stream is finished, finalises the message)

    async for chunk in chain.astream(message.content):
        await msg.stream_token(chunk.content)
    await msg.update()
