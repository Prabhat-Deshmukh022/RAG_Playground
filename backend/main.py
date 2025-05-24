import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import Optional
from rag_implementation import Basic_RAG,Hybrid_RAG
from typing import List

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
rag_pipelines={}

UPLOAD_DIR = "temp_uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

@app.post("/upload")
async def upload_pdf(option:int = Form(...), files: List[UploadFile] = File(...)):
    responses = []

    rag_pipeline = None


    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF")

            file_path = os.path.join(UPLOAD_DIR, file.filename)
            # option=int(option)

            match option:
                case 1:
                    rag_pipeline=Basic_RAG()
                
                case 2:
                    rag_pipeline=Hybrid_RAG()
                
                case _:
                    raise HTTPException(status_code=400, detail="Invalid option value")
            
            rag_pipelines[option]=rag_pipeline

            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest into RAG
            rag_pipeline.ingest(file_path)

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

    return {
        "status": "success",
        "results": responses
    }

@app.post("/query")
async def query_rag(option:int = Form(...), query: str = Form(...)):
    try:

        rag_pipeline = rag_pipelines.get(option)

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # option=int(option)

        # match option:
        #     case 1:
        #         rag_pipeline=Basic_RAG()
            
        #     case 2:
        #         rag_pipeline=Hybrid_RAG()

        #     case _:
        #         raise HTTPException(status_code=400, detail="Invalid option value")

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