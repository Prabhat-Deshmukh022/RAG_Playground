import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Optional
from rag_implementation import Basic_RAG  # import your BasicRAG class

app = FastAPI()

# Enable CORS if frontend is separate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init RAG engine
rag_pipeline = Basic_RAG()

UPLOAD_DIR = "temp_uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest into RAG
        rag_pipeline.ingest(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "File ingested successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure file is closed
        await file.close()

@app.post("/query")
async def query_rag(query: str = Form(...)):
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        result = rag_pipeline.send_query(query)
        
        if "error" in result and result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "status": "success",
            "answer": result.get("answer"),
            "contexts": result.get("contexts", [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))