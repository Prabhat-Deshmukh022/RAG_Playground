from typing import Dict, List, Optional

import requests

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.storage import StorageContext
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings

import shutil
from abc import ABC, abstractmethod

from backend.language_model_api import language_model_api

import os

class RAGEngine(ABC):

    def __init__(self,llm_option:int):
        self.text_splitter = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        self.documents=[]

        persist_dir = "./vector_db"
        os.makedirs(persist_dir, exist_ok=True)
        docstore_path = os.path.join(persist_dir, "docstore.json")

        # Only load from storage if docstore.json exists
        if os.path.exists(docstore_path):
            self.storage = StorageContext.from_defaults(persist_dir=persist_dir)
        else:
            self.storage = StorageContext.from_defaults()

        self.index=None

        self.llm_option=llm_option        

        Settings.embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # @abstractmethod
    def chunk_and_embed(self,file_paths:List[str]):
        input_docs=[]

        for file_path in file_paths:
            reader=SimpleDirectoryReader(input_files=[file_path])
            document=reader.load_data()
            input_docs.extend(document)
        
        self.documents.extend(input_docs)

        if not self.index:
            self.index=VectorStoreIndex.from_documents(self.documents,storage_context=self.storage)
        
        else:
            self.index.insert(self.documents)
        
        self.index.storage_context.persist()
    
    @abstractmethod
    def search(self,*args,**kwargs):
        pass

    def build_prompt(self, contexts: List[str], query: str) -> str:
        """
        Builds a RAG prompt with clear instructions and context separation.
        
        Args:
            contexts: List of context strings retrieved from search
            query: User's question/query
            
        Returns:
            Formatted prompt string
        """
        # Join contexts with clear separators
        context_str = "\n\n--- CONTEXT {} ---\n{}"
        numbered_contexts = [
            context_str.format(i+1, ctx.strip())
            for i, ctx in enumerate(contexts)
            if ctx.strip()  # Skip empty contexts
        ]
        
        prompt_template = (
            "You are a helpful AI assistant. Answer the question based only on the provided context. "
            "If the answer is not directly in the context, try to infer it from the information provided'."
            "{contexts}"
            "Question: {query}"
            "Answer:"
        )
        
        return prompt_template.format(
            contexts="".join(numbered_contexts),
            query=query.strip()
        )

    def send_query(self, query: str, top_k: int = 5) -> Dict[str, Optional[str]]:
        """
        Send query to Language model API with RAG context.
        
        Args:
            query: User's question
            top_k: Number of context passages to retrieve
            
        Returns:
            Dictionary containing:
            - answer: Generated response
            - contexts: Used context passages
            - error: Optional error message
        """
        try:
            # Retrieve relevant contexts
            search_results = self.search(query, top_k=top_k)
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "contexts": [],
                    "error": None
                }
                
            # Extract just the text if search returns (text, score) tuples
            contexts = [res[0] if isinstance(res, tuple) else res for res in search_results]
            
            # Build the prompt
            prompt = self.build_prompt(contexts, query)
            
            content = language_model_api(option=self.llm_option,prompt=prompt)
            
            return {
                # "answer": result.get("text", "").strip(),
                "answer":content,
                "contexts": contexts,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "answer": None,
                "contexts": contexts if 'contexts' in locals() else [],
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "answer": None,
                "contexts": contexts if 'contexts' in locals() else [],
                "error": f"Unexpected error: {str(e)}"
            }

    def clear_vector_db(self):
        persist_dir1='./storage'
        persist_dir2='./vector_db'

        if os.path.exists(persist_dir1):
            shutil.rmtree(persist_dir1)
        
        if os.path.exists(persist_dir2):
            shutil.rmtree(persist_dir2)

        self.index=None
        self.documents=[]
        self.storage=StorageContext.from_defaults()

