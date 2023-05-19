from fastapi import FastAPI ,File, UploadFile , Request, Form
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma   
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.agents import create_pandas_dataframe_agent
import os
import openai
import pandas as pd
from helper import AzureBlobStorage, AzureDatalakeStorage
from langchain.document_loaders import AzureBlobStorageFileLoader, SeleniumURLLoader, PyPDFLoader
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from supabase import create_client, Client
import re

load_dotenv()

class Question(BaseModel):
    question: str

class UrlInfo(BaseModel):
    url: str

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@app.post("/fileupload/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    # file_path = f"mydata/{file.filename}"
    # for filename in os.listdir("mydata"):
    #     os.remove(os.path.join("mydata", filename))
    # with open(file_path, "wb") as f:
    #         f.write(await file.read())
    storagehelper = AzureBlobStorage()
    u_name = await storagehelper.upload_file_to_directory(file.filename,file)
    
    global qa_global
    qa_global = create_qa(u_name)
    return templates.TemplateResponse("upload_result.html", {"request": request, "filename": file.filename})

@app.post("/fileuploadcsv/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    storagehelper = AzureBlobStorage()
    u_name = await storagehelper.upload_file_to_directory(file.filename,file)
    
    global csv_global
    csv_global = create_qa_csv(u_name)
    return templates.TemplateResponse("upload_result_csv.html", {"request": request, "filename": file.filename})

@app.post("/fileuploadpdf/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    storagehelper = AzureBlobStorage()
    u_name = await storagehelper.upload_file_to_directory(file.filename,file)
    
    global csv_global
    csv_global = create_qa_pdf(u_name)
    return templates.TemplateResponse("upload_result_pdf.html", {"request": request, "filename": file.filename})

@app.post("/fileuploadurl/")
async def urlembedding(request: Request,url: str = Form()):
    # Remove all existing files in the "files" directory
    print(url)
    global csv_global
    csv_global = create_qa_url(url)
    return templates.TemplateResponse("upload_result_url.html", {"request": request,"url": url})

@app.get("/files", response_class=HTMLResponse)
async def read_main(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("file_upload.html", {"request": request , "context": context})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # context = [
    #     {"name": "Alice", "age": 25},
    #     {"name": "Bob", "age": 30},
    # ]
    return templates.TemplateResponse("login.html",{"request": request})


@app.get("/db", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("pdf_upload_db.html", {"request": request , "context": context})

@app.post("/dbqna/")
async def file_upload_db(request: Request,file: UploadFile = File(...)):
    return templates.TemplateResponse("upload_result_db.html", {"request": request, "filename": file.filename})



def create_qa(filename:str) -> RetrievalQA:
    # Create a language model for Q&A
    llm = AzureOpenAI(deployment_name="text-davinci-003")

    # Create embeddings for text documents
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    loader = AzureBlobStorageFileLoader(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"), container='testcontainer', blob_name=filename)
    # Load text documents
    # loader = DirectoryLoader('mydata', glob="**/*.txt")
    documents = loader.load()

    # Split text documents into chunks 중지 시퀀스
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create a vector store for text documents
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a retrieval-based Q&A system
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    return qa

def create_qa_csv(filename: str) -> create_pandas_dataframe_agent:

  """This function creates a language model for Q&A and returns a Pandas DataFrame agent.

  Args:
    filename: The name of the CSV file to read.

  Returns:
    A Pandas DataFrame agent.
  """

  # Create a language model for Q&A
  llm = AzureOpenAI(deployment_name="text-davinci-003")

  # Create a BlobServiceClient object
  service_client = BlobServiceClient.from_connection_string(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

  # Create a BlobClient object
  blob_client = service_client.get_blob_client('testcontainer', filename)

  # Download the CSV file from Azure Blob Storage
  with open(filename, "wb") as my_blob:
    blob_data = blob_client.download_blob()
    blob_data.readinto(my_blob)

  # Read the CSV file into a Pandas DataFrame
  df = pd.read_csv(filename)

  # Delete the CSV file
  os.remove(filename)

  # Create a Pandas DataFrame agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True)

  return agent

def create_qa_pdf(filename:str) -> RetrievalQA:
    # Create a language model for Q&A
    llm = AzureOpenAI(deployment_name="text-davinci-003")

    # Create embeddings for text documents
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

      # Create a BlobServiceClient object
    service_client = BlobServiceClient.from_connection_string(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    # Load text documents
    # loader = DirectoryLoader('mydata', glob="**/*.txt")
  # Create a BlobClient object
    blob_client = service_client.get_blob_client('testcontainer', filename)

  # Download the CSV file from Azure Blob Storage
    with open(filename, "wb") as my_blob:
      blob_data = blob_client.download_blob()
      blob_data.readinto(my_blob)

    loader = PyPDFLoader(filename)
    texts = loader.load_and_split()

    # Delete the CSV file
    os.remove(filename)
    # Create a vector store for text documents
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a retrieval-based Q&A system
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    return qa
  
def create_qa_url(url:str) -> RetrievalQA:
    # Create a language model for Q&A
    llm = AzureOpenAI(deployment_name="text-davinci-003")

    # Create embeddings for text documents
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    urls = [url]
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()
    
    # Split text documents into chunks 중지 시퀀스
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    # Create a vector store for text documents
    docsearch = Chroma.from_documents(texts, embeddings)
    

    # Create a retrieval-based Q&A system
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    return qa

# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
# pages = loader.load_and_split()

@app.post("/qna/")
def get_qna(question: Question):
    # qa = qa_global
    answer = qa_global.run(question.question)
    return {"data": answer}

@app.post("/qnacsv/")
def get_qna(question: Question):
    # qa = qa_global
    answer = csv_global.run(question.question)
    return {"data": answer}

@app.post("/qnapdf/")
def get_qna(question: Question):
    # qa = qa_global
    answer = csv_global.run(question.question)
    return {"data": answer, "data_ko":answer}

@app.post("/qnadb/")
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