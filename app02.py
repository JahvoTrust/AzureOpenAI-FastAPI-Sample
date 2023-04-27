from fastapi import FastAPI ,File, UploadFile
from typing import Optional
from pydantic import BaseModel
import os


app = FastAPI()

@app.get("/")
def index():
    return {"name":"Second Data"}

@app.post("/fileupload/")
async def file_upload(file: UploadFile = File(...)):
    # Remove all existing files in the "files" directory
  
    file_path = f"mydata/{file.filename}"
    for filename in os.listdir("mydata"):
        os.remove(os.path.join("mydata", filename))
    with open(file_path, "wb") as f:
            f.write(await file.read())
    return {"filename": file.filename}
  