#from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_docling.loader import DoclingLoader
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import hashlib

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError

# 1. Read documents
# 2. Extract pages
# 3. Decide how to chunk/split text
# 4. Embed using nomad
# 5. Store in Chroma DB.

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True
            )
        )
    }
)

def generate_chunk_hash(chunk, index):
    source = chunk.metadata.get("source")
    pages = chunk.metadata.get("pages")
    content = chunk.page_content
    chunk_id = hashlib.sha1(f"{source}|{pages}|{index}|{content}".encode()).hexdigest()
    return chunk_id

def prepare_chunk(chunk, index):
    chunk.metadata = simplify_docling_metadata(chunk.metadata)
    chunk.metadata["chunk_hash"] = generate_chunk_hash(chunk, index)
    chunk.metadata["chunk_index"] = index
    return chunk


def ingest(directory: Path, vector_db: Chroma) -> None:
    print(f"Loading documents from path {directory}...")
    # Load in the documents
    pdf_files = directory.rglob("*.pdf")
    #csv_files = directory.rglob("*.csv")
    files_found = 0

    for file in pdf_files:
        print(f"Found pdf file: {file.name}")
        files_found += 1
        loader = DoclingLoader(file, converter=converter)
        chunks_loaded = 0
        chunks = []

        try:
            # Chunk the file
            for idx, chunk in enumerate(loader.lazy_load()):
                
                # Prepare this chunk
                prepared_chunk = prepare_chunk(chunk, idx)
                chunks.append(prepared_chunk)

                # If this is a big file, we don't want chunks getting too big, so stop, prepare and embed
                if len(chunks) >= 100:
                    chunks_loaded += len(chunks)
                    ids = [chunk.metadata["chunk_hash"] for chunk in chunks]
                    vector_db.add_documents(documents=chunks, ids=ids)
                    chunks.clear()
        except ConversionError as error:
            print(f"Docling failed for {file}: {error}")
            chunks.clear()
            continue

        # Embed any remaining chunks
        if len(chunks) > 0:
            chunks_loaded += len(chunks)
            ids = [chunk.metadata["chunk_hash"] for chunk in chunks]
            vector_db.add_documents(documents=chunks, ids=ids)

        print(f"{chunks_loaded} chunks loaded from document {file.name}.")

    print(f"{files_found} files processed.")

def simplify_docling_metadata(metadata: dict) -> dict:
    source = str(metadata.get("source", ""))
    dl_meta = metadata.get("dl_meta", {})
    origin = dl_meta.get("origin", {})
    headings = dl_meta.get("headings", [])
    doc_items = dl_meta.get("doc_items", [])

    pages = set()

    for item in doc_items:
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                pages.add(page_no)

    return {
        "source": source,
        "filename": str(origin.get("filename") or Path(source).name),
        "pages": ",".join(str(page) for page in sorted(pages)),
        "headings": " > ".join(headings),
        "mimetype": str(origin.get("mimetype", "")),
    }


# Set document paths
shared_path = Path(r"Y:\Documents")
josh_path = Path(r"Z:\Documents")

# Define Embedding model
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# Define Vector DB which takes in an Embedding Model
vector_db = Chroma(
    collection_name="home_llm",
    embedding_function=embed_model,
    persist_directory="chroma_db"
)

# Ingest
ingest(shared_path, vector_db)
ingest(josh_path, vector_db)




