from fastapi import HTTPException
from rag_implementation import Basic_RAG, Hybrid_RAG, AutoMerge_RAG

class RAG_Orchestrator:

    def __init__(self,llm_option:str,rag_option:str):
        
        self.llm_option=llm_option
        self.rag_option=rag_option
    
    def orchestrate(self):

        match self.rag_option:
            case 1:
                print("BASIC RAG")
                return Basic_RAG(self.llm_option)
            case 2:
                print("HYBRID RAG")
                return Hybrid_RAG(self.llm_option)
            case 3:
                print("AUTO RAG")
                return AutoMerge_RAG(self.llm_option)
            case _:
                raise HTTPException(status_code=400, detail="Invalid option value")
