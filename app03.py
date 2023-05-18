from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import openai
from supabase import create_client, Client
import tiktoken

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

class Essay(BaseModel):
    title: str 
    url: str 
    date: str 
    thanks: str 
    content: str 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "-", "age": 0},
        {"name": "-", "age": 0},
    ]
    return templates.TemplateResponse("supabase.html", {"request": request , "context": context})

@app.get("/supabase/{query}")
async def get_data(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    embedding = embeddings.embed_query(query)

    response = supabase.rpc("pg_search", {"query_embedding": embedding, "similarity_threshold": 0.7, "match_count": 3}).execute()

    return response

@app.post("/insert")
async def insert_data(essay: Essay):

    print(essay.title, essay.url, essay.date, essay.thanks, essay.content)

    # id컬럼 max값 
    max_id = supabase.table('pg').select('id').order('id', desc='false').limit(1).execute()

    # content 토큰 수
    tokenizer = tiktoken.get_encoding("cl100k_base")

    input = essay.content

    sample_encode = tokenizer.encode(input) 
    decode = tokenizer.decode_tokens_bytes(sample_encode)
    token_len = len(decode)

    id = max_id.data[0]['id'] + 1
    content_len = len(essay.content)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding = embeddings.embed_query(essay.content)

    response = supabase.table('pg').insert({"id": id, "essay_title": essay.title, "essay_url": essay.url, "essay_date": essay.date, "essay_thanks": essay.thanks, "content": essay.content, "content_length": content_len, "content_tokens": token_len, "embedding": embedding}).execute()

    return {"essay_title": essay.title, "essay_url": essay.url, "essay_date": essay.date, "essay_thanks": essay.thanks, "content": essay.content}