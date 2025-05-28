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
from rag_orchestrator import RAG_Orchestrator

app = FastAPI(
    title="RAG Architecture Comparison API",
    description="This endpoint allows uploading PDFs and comparing different RAG pipelines (Basic, Hybrid, Auto-Merge)."
)

app.state.rag_pipelines={}

# Enable CORS if frontend is separate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/upload", 
          summary="Uploading multiple pdfs", 
          description="The endpoint accepts upto three pdfs and ingests them according to the user's chosen RAG architecture")
async def upload_pdf(rag_option:int = Form(...), llm_option:int=Form(...),files: List[UploadFile] = File(...)):
    responses = []
    local_files=[]

    rag_pipeline=RAG_Orchestrator(llm_option=llm_option,rag_option=rag_option).orchestrate()
    app.state.rag_pipelines[(rag_option)] = rag_pipeline

    print(files)

    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF")

            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest into the SAME pipeline
            # rag_pipeline.chunk(file_path=file_path)

            local_files.append(file_path)

        except Exception as e:
            raise HTTPException(status_code=400,detail=f"ERROR: {str(e)}")
        
        finally:
            await file.close()
    
    try:
        rag_pipeline.chunk_and_embed(file_paths=local_files)
        responses.append({
                "filename": local_files,
                "status": "ingested"
            })
    
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"ERROR: {str(e)}")

    
    clear_directory(UPLOAD_DIR)
    # rag_pipeline.ingest()
    # Store the pipeline after all files are ingested
    # rag_pipelines[rag_option] = rag_pipeline

    return {
        "status": "success",
        "results": responses
    }

@app.post("/query",
          summary="Talking to the language model")
async def query_rag(rag_option:int, query: str = Form(...)):
    try:

        print(query)

        rag_pipeline = app.state.rag_pipelines.get((rag_option))

        if rag_pipeline is None:
            raise HTTPException(status_code=400, detail="No pipeline loaded for this option. Please upload documents first.") 

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = rag_pipeline.send_query(query)

        # result = rag_pipeline.s
        
        if "error" in result and result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "status": "success",
            "answer": result.get("answer"),
            "contexts": result.get("contexts", [])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

