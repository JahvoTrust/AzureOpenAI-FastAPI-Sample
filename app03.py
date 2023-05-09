from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os
import openai
from supabase import create_client, Client

load_dotenv()

openai.api_version = '2022-12-01'
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = 'azure'
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_id = 'text-embedding-ada-002'

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "-", "age": 0},
        {"name": "-", "age": 0},
    ]
    return templates.TemplateResponse("supabase.html", {"request": request , "context": context})

@app.get("/supabase/{query}")
async def get_data(query):
    embeddings = openai.Embedding.create(deployment_id=deployment_id, input=query)
    embedding = embeddings['data'][0]['embedding']

    response = supabase.rpc("pg_search", {"query_embedding": embedding, "similarity_threshold": 0.7, "match_count": 1}).execute()

    return response
