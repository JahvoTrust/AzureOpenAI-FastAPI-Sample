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
import os
import openai

load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

templates = Jinja2Templates(directory="templates")


class Question(BaseModel):
    question: str

def create_qa() -> RetrievalQA:
    # Create a language model for Q&A
    llm = AzureOpenAI(deployment_name="text-davinci-003")

    # Create embeddings for text documents
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    # Load text documents
    loader = DirectoryLoader('mydata', glob="**/*.txt")
    documents = loader.load()

    # Split text documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    
    # Create a vector store for text documents
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a retrieval-based Q&A system
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    return qa

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("index.html", {"request": request , "context": context})
  

@app.post("/fileupload/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    file_path = f"mydata/{file.filename}"
    for filename in os.listdir("mydata"):
        os.remove(os.path.join("mydata", filename))
    with open(file_path, "wb") as f:
            f.write(await file.read())
    return templates.TemplateResponse("upload_result.html", {"request": request, "filename": file.filename})

@app.post("/qna/")
def get_qna(question: Question):
    qa = create_qa()
    answer = qa.run(question.question)
    return {"data": answer}

# @app.post("/qna/", response_class=HTMLResponse)
# async def get_qna(request: Request, question: Question):
#     qa = create_qa()
#     answer = qa.run(question.question)
#     return templates.TemplateResponse("index.html", {"request": request, "answer": answer})

@app.post("/fileupload/")
async def file_upload(file: UploadFile = File(...)):
    for filename in os.listdir("mydata"):
        os.remove(os.path.join("mydata", filename))
    
    file_path = f"mydata/{file.filename}"
    with open(file_path, "wb") as f:
            f.write(await file.read())
    return {"filename": file.filename}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
