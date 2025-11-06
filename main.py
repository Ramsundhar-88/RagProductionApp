import logging  # logging log or status 
import os  # read the env variable , file path 
import uuid  # create unique ids for the pdf 
import datetime  # normal timestamp 

from fastapi import FastAPI  # just like express or nodejs 
import inngest 
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGQueryResult, RAGUpsertResult

load_dotenv()

# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# Initialize Qdrant client globally
qdrant_storage = QdrantStorage()


@inngest_client.create_function(
    fn_id="RAG Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):

    # Step 1: Load and chunk the PDF
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    # Step 2: Embed and upsert to Qdrant
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Embed the chunks
        vecs = embed_texts(chunks)

        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]

        # Prepare payloads
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        # Upsert into Qdrant
        qdrant_storage.upsert(ids, vecs, payloads)

        # Return result info
        return RAGUpsertResult(ingested=len(chunks), source_id=source_id)

    # Run both steps sequentially in Inngest pipeline
    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


# Initialize FastAPI app
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf])
