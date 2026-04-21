from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_documents(directory: Path) -> list:
    print(f"Loading documents from path {directory}...")
    # Load in a document
    pages = []
    docs_found = 0
    pdf_files = directory.rglob("*.pdf")
    csv_files = directory.rglob("*.csv")

    for file in pdf_files:
        print(f"Found pdf file: {file.name}")
        loader = PyPDFLoader(file)
        pages.extend(loader.load())
        docs_found += 1

    for file in csv_files:
        print(f"Found csv file: {file.name}")
        loader = CSVLoader(file)
        pages.extend(loader.load())
        docs_found += 1
    print(f"{len(pages)} pages loaded in from {docs_found} documents.\n")
    return pages

def chunk_documents(docs: list, chunk_size: int, chunk_overlap: float) -> list:
    # Split it into chunks
    # this is in characters, not tokens.  Research states to start with 15% overlap and 256 tokens for this
    # since there's 3/4 characters in a token, multiply 256 by 4 roughtyly.  15% of 1000 is 150 for overlap. 
    chunk_size *= 4 # * 4 for the character equivilant
    chunk_overlap_chars = int(chunk_size * chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap_chars) 
    splits = text_splitter.split_documents(docs)
    
    # Add Ids to the chunks for chroma to use to not duplicate chunks
    for i, chunk in enumerate(splits):
        chunk.metadata["chunk_index"] = i
    print(f"Split documents into {len(splits)}, {chunk_size} character long chunks.\n")
    return splits

def embed_splits(splits: list, vector_db: Chroma) -> None:
    ids = [f"{chunk.metadata['source']}:{chunk.metadata['page']}:{chunk.metadata['chunk_index']}" for chunk in splits]
    print("Chunks being embedded into Chroma DB...")
    vector_db.add_documents(documents=splits, ids=ids)
    print("Chunks embedded into Chroma DB.")

# Load documents
shared_path = Path("Y:\Documents")
josh_path = Path("Z:\Documents")
shared_docs = load_documents(shared_path)
josh_docs = load_documents(josh_path)

shared_docs.extend(josh_docs)
full_docs = shared_docs

# Split the documents
splits = chunk_documents(full_docs, chunk_size=256, chunk_overlap=0.15)

# Embed 
embed_function = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    collection_name="home_llm",
    embedding_function=embed_function,
    persist_directory="chroma_db"
)
embed_splits(splits, vector_db=vector_db)




