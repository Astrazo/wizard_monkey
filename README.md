# Wizard Monkey

A personal, locally hosted AI home assistant with the modest eventual goal of
replacing Google Home. A never-ending project, assuming I eventually find the
time.

The name is a homage to the Wizard Monkey from the Bloons TD games, because he is cool.

## What it does

Wizard Monkey currently provides a local RAG system for asking questions about
personal documents.

PDFs are processed, embedded, and stored in a local vector database. The chat
interface determines when a question requires information from those documents,
retrieves the relevant context, and uses it to generate an answer.

Conversation history provides context for follow-up questions, so asking
"what about the other one?" does not require starting the interrogation again.

For now, this makes Wizard Monkey primarily a personal document assistant. The
intention is for that to eventually become just one part of a much broader home
assistant.

## Tech

The project is built in Python using:

- **LangChain** for LLM and retrieval orchestration
- **Chainlit** for the chat interface
- **Ollama** for locally hosted language models
- **nomic-embed-text** for local document embeddings
- **Docling** for PDF processing
- **Chroma** for local vector storage

## Status

Google Home is evidently more than a RAG loop.  So planned areas of improvement include:

- **Voice interaction** using Whisper for speech-to-text and Piper for
  text-to-speech
- **Home automation** through deterministic tools and integrations
- **Additional personal knowledge sources** beyond PDFs
- **Persistent conversation and user context**
- **Tailscale** for secure access outside the home network
- Running the complete system from a dedicated home server

There is deliberately no real definition of "finished" here. If it works and I
think of something else that would be useful, it just means Wizard Monkey now has
another feature on its roadmap.