from fastapi import FastAPI, Request, File, UploadFile 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import os
from supabase import create_client, Client
from langchain.document_loaders import PyPDFLoader
from langchain.llms import AzureOpenAI

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


