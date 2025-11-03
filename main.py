import logging #looging log or status 
import os # read the env variable , file path 
import uuid # it create unique ids for the pdf 
import datetime # normal timestamp 

from fastapi import FastAPI # just like express or nodejs 
import inngest 
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")  # Changed 'triggers' to 'trigger'
)
async def rag_inngest_pdf(ctx: inngest.Context):
    return {"hello": "world"}


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf])  # Added the function to the list