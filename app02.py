from fastapi import FastAPI ,File, UploadFile , Request
from typing import Optional
from pydantic import BaseModel
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.post("/fileupload/")
async def file_upload(request: Request,file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    file_path = f"mydata/{file.filename}"
    for filename in os.listdir("mydata"):
        os.remove(os.path.join("mydata", filename))
    with open(file_path, "wb") as f:
            f.write(await file.read())
    return templates.TemplateResponse("upload_result.html", {"request": request, "filename": file.filename})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 30},
    ]
    return templates.TemplateResponse("index.html", {"request": request , "context": context})
  