from fastapi import FastAPI, UploadFile, File
import os
import shutil

app = FastAPI(title="AURA Doc Service")

UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/health")
def health_check():
    return {
        "status": "AURA doc-service is running",
        "service": "doc-service"
    }

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "message": "File uploaded successfully"
    }
