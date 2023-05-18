from fastapi import FastAPI, Request, File, UploadFile 
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from helper import AzureDatalakeStorage
from langchain.document_loaders import AzureBlobStorageFileLoader
import openai 
import os
import pandas as pd
import re

load_dotenv()


openai.api_version = '2022-12-01'
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = 'azure'
openai.api_key = os.getenv("OPENAI_API_KEY")

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("pdf_upload.html", {"request": request , "context": context})

@app.post("/fileuploadpdf/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    storagehelper = AzureDatalakeStorage()
    await storagehelper.upload_file_to_directory("pdf", file.filename,file)
    
    global pdf_global
    pdf_global = create_qa_pdf(file.filename)
    return templates.TemplateResponse("upload_result_pdf.html", {"request": request, "filename": file.filename})


def create_qa_pdf(filename:str) -> RetrievalQA:
    # Create a language model for Q&A
    # llm = AzureOpenAI(deployment_name="text-davinci-003")

    # Create embeddings for text documents
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    loader = AzureBlobStorageFileLoader(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"), container='data/pdf', blob_name=filename)

    documents = loader.load()

    # Split text documents into chunks 중지 시퀀스
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    list_content = []
    list_embedding = []

    for i in range(len(texts)):
        list_content.append(texts[i].page_content)

        # embedding = openai.Embedding.create(deployment_id="text-embedding-ada-002", input=texts[i].page_content)
        # list_embedding.append(embedding['data'][0]['embedding'])
        embedding = embeddings.embed_query(texts[i].page_content)
        list_embedding.append(embedding)

    # list_content와 list_embedding을 사용하여 DataFrame 생성
    df = pd.DataFrame(zip(list_content, list_embedding))
    df.columns = ['content', 'embedding']

    # supabase의 'file' 테이블에 데이터 삽입
    for i in range(len(df)): 
        response = supabase.table('file').insert({"name": filename, "content": df['content'][i], "embedding":df['embedding'][i]}).execute()

    return response

@app.post("/qnapdf/")
def get_qna(question: Question):
    # 질문 임베딩 
    print(question.question)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    embedding = embeddings.embed_query(question.question)
    response = supabase.rpc("match_file", {"query_embedding": embedding, "match_threshold": 0.7, "match_count": 2}).execute()

    contents = []

    for i in range(len(response.data)):
        contents.append(response.data[i]['content'])

    text = ' '.join(contents)
    new_text = re.sub(r"\s+|\n", " ", text)

    header = """Answer the question truthfully using document, if unsure, say "잘 모르겠습니다"\n\nDocument:\n"""
    prompt = header + new_text + "\n\n Q: " + question.question + "\n A: "

    print(prompt)
    completion = openai.Completion.create(deployment_id="text-davinci-003",
                                        prompt=prompt, temperature=0, top_p=1.0, max_tokens=2000)
    print(completion['choices'][0]['text'])
    
    return {"data": completion['choices'][0]['text']}