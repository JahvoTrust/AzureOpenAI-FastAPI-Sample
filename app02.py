from fastapi import FastAPI ,File, UploadFile , Request
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
from helper import AzureBlobStorage
from langchain.document_loaders import AzureBlobStorageFileLoader, CSVLoader
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("file_upload.html", {"request": request , "context": context})

class Question(BaseModel):
    question: str

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

def create_qa_csv(filename : str) -> create_pandas_dataframe_agent:
    # Create a language model for Q&A
    llm = AzureOpenAI(deployment_name="text-davinci-003")
    # file_path = f"mydata/{filename}"
    # df = pd.read_csv(file_path)

      # Create a BlobServiceClient object
    service_client = BlobServiceClient.from_connection_string(conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    
    # Create a BlobClient object
    blob_client = service_client.get_blob_client('testcontainer', filename)
    
    with open(filename, "wb") as my_blob:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_blob)
   
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename)

    # Use the os.remove() function to delete the file
    os.remove(filename)

    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    return agent

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

