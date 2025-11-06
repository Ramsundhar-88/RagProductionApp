from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from typing import List

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client (requires OPENAI_API_KEY in .env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define embedding model and dimensions
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Initialize text splitter
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> List[str]:
    """
    Load a PDF file and split it into text chunks for embedding.
    """
    # Read PDF contents
    docs = PDFReader().load_data(file=path)

    # Extract text safely
    texts = [d.text for d in docs if hasattr(d, "text") and d.text]

    # Split into smaller chunks
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks using OpenAI embeddings.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]
