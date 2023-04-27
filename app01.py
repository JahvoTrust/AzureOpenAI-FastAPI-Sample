from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma   
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


students = {
    1: {
    "name" : "Jhon",
    "age" : 22,
    "gender" : "male"
    }
}

class Student(BaseModel):
    name : str
    age : int
    gender : str


@app.get("/")
def index():
    return {"name":"First Data"}

@app.get("/getstudent/{student_id}")
def getstudent(student_id : int):
    return students[student_id]

@app.get("/get-student-name/")
def getbyname(*,name : Optional[str] = None , test : Optional[int] = None):
    for student_id in students:
        if students[student_id]["name"] == name:
            return students[student_id]
    return {"Data":"Not Found"}

@app.post("/register-student/{student_id}")
def register_student(student_id : int , student : Student):
    students[student_id] = student
    return students[student_id] 

@app.get("/get-embeddings/")
def get_embeddings():
    # Azure OpenAI model mapping
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    text = "Algoritma is a data science school based in Indonesia and Supertype is a data science consultancy with a distributed team of data and analytics engineers."
    doc_embeddings = embeddings.embed_documents([text])
    return {"data": doc_embeddings}



def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

 # Create a completion - qna를 위하여 davinci 모델생성
llm = AzureOpenAI(deployment_name="text-davinci-003")

    # text embedding 을 위해서는 좀더 저렴한 ada 모델을 사용
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    # embeddings = OpenAIEmbeddings()

    # loader = TextLoader('news/summary.txt')
loader = DirectoryLoader('mydata', glob="**/*.txt")

documents = loader.load()
    # print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
        # llm=OpenAI(), 
        llm=llm, 
        chain_type="stuff", 
        # retriever=docsearch.as_retriever()
        retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

@app.get("/qna/")
def get_qna(question : str):

    # template = """
    # I need your expertise as a marketing consultant for a new product launch.
    # Here are some examples of successful product names:
    # wearable fitness tracker, Fitbit
    # premium headphones, Beats
    # ride-sharing app, Uber
    # The name should be unique, memorable and relevant to the product.

    # What is a good name for a {product_type} that offers {benefit}?
    # """

    # prompt = PromptTemplate(
    #     input_variables=["product_type", "benefit"],
    #     template=template,
    # )
   
    # print(llm(
    #     prompt.format(
    #     product_type="pair of sunglasses",
    #     benefit = 'high altitude protection'
    # )
    # ))
   
    answer = qa.run(question)
    return {"data": answer}