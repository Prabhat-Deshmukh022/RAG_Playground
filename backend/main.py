import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Optional
from rag_implementation import Basic_RAG,Hybrid_RAG,AutoMerge_RAG
from typing import List

from remove_file_contents import clear_directory

app = FastAPI(
    title="RAG Architecture Comparison API",
    description="This endpoint allows uploading PDFs and comparing different RAG pipelines (Basic, Hybrid, Auto-Merge)."
)

# Enable CORS if frontend is separate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init RAG engine
rag_pipelines={}

UPLOAD_DIR = "temp_uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

@app.post("/upload", 
          summary="Uploading multiple pdfs", 
          description="The endpoint accepts upto three pdfs and ingests them according to the user's chosen RAG architecture")
async def upload_pdf(rag_option:int = Form(...), llm_option:int=Form(...),files: List[UploadFile] = File(...)):
    responses = []

    # Instantiate the pipeline ONCE
    match rag_option:
        case 1:
            rag_pipeline = Basic_RAG(option=llm_option)
        case 2:
            rag_pipeline = Hybrid_RAG(option=llm_option)
        case 3:
            rag_pipeline = AutoMerge_RAG(option=llm_option)
        case _:
            raise HTTPException(status_code=400, detail="Invalid option value")

    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF")

            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest into the SAME pipeline
            rag_pipeline.chunk(file_path=file_path)

            responses.append({
                "filename": file.filename,
                "status": "ingested"
            })

        except Exception as e:
            responses.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
        finally:
            await file.close()
    
    clear_directory(UPLOAD_DIR)
    rag_pipeline.ingest()
    # Store the pipeline after all files are ingested
    rag_pipelines[rag_option] = rag_pipeline

    return {
        "status": "success",
        "results": responses
    }

@app.post("/query",
          summary="Talking to the language model")
async def query_rag(option:int, query: str = Form(...)):
    try:

        print(query)

        rag_pipeline = rag_pipelines.get(option)

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

