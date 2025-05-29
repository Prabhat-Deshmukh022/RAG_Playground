import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG.RAG_Engine import RAGEngine

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers import EnsembleRetriever
# # from langchain_community.retrievers import BM25Retriever
# from langchain.docstore.document import Document
# from sentence_transformers import SentenceTransformer

# from llama_index.core import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     Settings
# )
# from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever 
from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.schema import Document
# from llama_index.core.node_parser import HierarchicalNodeParser

from typing import List, Optional, Dict
# import numpy as np
# import faiss
# import requests

# from remove_file_contents import clear_directory
# from language_model_api import language_model_api
from dotenv import load_dotenv
# from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = 'temp_uploads'


class Basic_RAG(RAGEngine):
    def __init__(self, llm_option):
        super().__init__(llm_option)
    
    def search(self, query:str, top_k:int=10):
        try:
            retriever = VectorIndexRetriever(self.index,similarity_top_k=top_k)
            results = retriever.retrieve(query)
            return [(node.get_content(), node.score) for node in results]
        
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return None

class Hybrid_RAG(RAGEngine):
    def __init__(self, llm_option):
        super().__init__(llm_option)
    
    def search(self, query:str, top_k:int=5):
        vector_retriever=VectorIndexRetriever(self.index,similarity_top_k=top_k)
        bm25_retriever=BM25Retriever.from_defaults(self.index,similarity_top_k=top_k)

        try:
            vector_results = vector_retriever.retrieve(query)
            bm25_results = bm25_retriever.retrieve(query)

            # print(vector_results,bm25_results)

            print(f"LENGTH OF BM25 {len(bm25_results)}")

            seen=set()
            combined=[]

            combined.extend(vector_results+bm25_results)
            combined.sort(key=lambda x:x.score, reverse=True)
            
            return [(node.get_content(), node.score) for node in combined]

    
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return None

class AutoMerge_RAG(RAGEngine):
    
    def __init__(self, llm_option):
        super().__init__(llm_option)
    
    def search(self, query: str, top_k: int = 10) -> Optional[List[str]]:
        if not self.index:
            print("Index not initialized.")
            return None
        try:
            # retriever = AutoMergingRetriever(index=self.index, similarity_top_k=top_k)
            vector_index_retriever=VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
            storage_context = self.index.storage_context
            retriever = AutoMergingRetriever(vector_retriever=vector_index_retriever,storage_context=storage_context)
            results = retriever.retrieve(query)
            return [(node.get_content(), node.score) for node in results]
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return None



# class Basic_RAG(RAGEngine):

#     def __init__(self, option:int):
#         self.text_splitter = SentenceTransformersTokenTextSplitter(
#             chunk_overlap=50,
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             chunk_size=512
#         )

#         self.model = SentenceTransformer("all-MiniLM-L6-v2")

#         self.index=None
#         self.chunks=[]
#         self.dim=None

#         self.option=option

#     def chunk(self, file_path:str) -> List[str]:
#         loader = PyPDFLoader(file_path=file_path,mode='page')
#         chunks = loader.load_and_split(text_splitter=self.text_splitter)

#         final_chunks = [chunk.page_content for chunk in chunks]

#         self.chunks.extend(final_chunks)

#         return final_chunks
        
#     def embed(self, chunks:list[str], batch_size:int = 32) -> np.ndarray:
#         embeddings = self.model.encode(
#             chunks,
#             batch_size=batch_size,
#             show_progress_bar=False,
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         )

#         return embeddings
    
#     def ingest(self) -> bool:
#         try:
#             # self.chunks.extend(self.chunk(file_path=file_path))

#             if not self.chunks:
#                 raise ValueError("No valid chunks extracted!")
            
#             print(f"Length of chunks {len(self.chunks)}")
#             vectors = self.embed(chunks=self.chunks)
#             print(f"Shape of vector {vectors.shape}")
#             self.dim = vectors.shape[1]

#             # faiss.normalize_L2(vectors)

#             self.index = faiss.IndexFlatIP(self.dim)
#             # faiss.IndexIn

#             self.index.add(vectors)

#             return True
        
#         except Exception as e:
#             print(f"Ingestion failed: {str(e)}")
#             self.index = None
#             self.chunks = []
#             return False

#     def search(self, query:str, top_k:int = 10) -> Optional[List[str]]:
#         if not self.index or not self.chunks:
#             print("Index not initialized - please ingest documents first")
#             # return {"message":"Error"}
#             return None
            
#         try:
#             # Embed and normalize query
#             query_vector = self.embed([query])
#             # faiss.normalize_L2(query_vector)
            
#             # Search with score thresholding
#             distances, indices = self.index.search(query_vector,k=top_k)

#             print(f"Distances {distances[0]} and indices {indices}")
            
#             # Convert FAISS inner product to cosine similarity (0-1)
#             # scores = [max(0, 1 - distance) for distance in distances[0]]
            
#             # Return chunks with their scores
#             results = [
#                 (self.chunks[idx], float(score))
#                 for idx, score in zip(indices[0], distances[0])
#                 if idx >= 0  # FAISS returns -1 for invalid indices
#             ]

#             # print("Retrieved contexts: ",results)
#             return results
            
#         except Exception as e:
#             print(f"Search failed: {str(e)}")
#             return None

#     def build_prompt(self, contexts: List[str], query: str) -> str:
#         """
#         Builds a RAG prompt with clear instructions and context separation.
        
#         Args:
#             contexts: List of context strings retrieved from search
#             query: User's question/query
            
#         Returns:
#             Formatted prompt string
#         """
#         # Join contexts with clear separators
#         context_str = "\n\n--- CONTEXT {} ---\n{}"
#         numbered_contexts = [
#             context_str.format(i+1, ctx.strip())
#             for i, ctx in enumerate(contexts)
#             if ctx.strip()  # Skip empty contexts
#         ]
        
#         prompt_template = (
#             "You are a helpful AI assistant. Answer the question based only on the provided context. "
#             "If the answer is not directly in the context, try to infer it from the information provided'.\n\n"
#             # "Continue your answer after the ':' puncuation"
#             "{contexts}\n\n"
#             "Question: {query}\n"
#             "Answer:"
#         )
        
#         prompt = prompt_template.format(
#             contexts="\n\n".join(numbered_contexts),
#             query=query.strip()
#         )
#         # print("Returned prompt: ",prompt)
#         return prompt

#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=4, max=10),
#         reraise=True
#     )

#     def send_query(self, query: str, top_k: int = 10) -> Dict[str, Optional[str]]:
#         """
#         Send query to Gemini API with RAG context.
        
#         Args:
#             query: User's question
#             top_k: Number of context passages to retrieve
            
#         Returns:
#             Dictionary containing:
#             - answer: Generated response
#             - contexts: Used context passages
#             - error: Optional error message
#         """
#         try:
#             # Retrieve relevant contexts
#             search_results = self.search(query, top_k=top_k)
#             if not search_results:
#                 return {
#                     "answer": "I couldn't find any relevant information to answer your question.",
#                     "contexts": [],
#                     "error": None
#                 }
                
#             # Extract just the text if search returns (text, score) tuples
#             contexts = [res[0] if isinstance(res, tuple) else res for res in search_results]
            
#             # Build the prompt
#             prompt = self.build_prompt(contexts, query)
            
#             result = language_model_api(option=self.option,prompt=prompt)
            
#             return {
#                 # "answer": result.get("text", "").strip(),
#                 "answer":result,
#                 "contexts": contexts,
#                 "error": None
#             }
            
#         except requests.exceptions.RequestException as e:
#             return {
#                 "answer": None,
#                 "contexts": contexts if 'contexts' in locals() else [],
#                 "error": f"API request failed: {str(e)}"
#             }
#         except Exception as e:
#             return {
#                 "answer": None,
#                 "contexts": contexts if 'contexts' in locals() else [],
#                 "error": f"Unexpected error: {str(e)}"
#             }

# class Hybrid_RAG(RAGEngine):
#     def __init__(self,option:int):
#         # self.model = "mistral-small"
#         self.temperature = 0.0

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=50,
#             length_function=len,
#             separators=["\n\n","\n"," "]
#         )

#         self.embedding_model = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.retriever = None
#         self.documents = []
#         # self.api_key=MISTRAL_API_KEY
#         self.option=option
#         # self.api_url = "https://api.mistral.ai/v1/chat/completions"
    
#     def chunk(self, file_path: str) -> List[str]:
#         loader = PyPDFLoader(file_path=file_path)
#         documents = loader.load()

#         chunks = self.text_splitter.split_documents(documents=documents)
#         self.documents.extend(chunks)

#         # print(chunks)

#         # clear_directory(UPLOAD_DIR)

#         return [chunk.page_content for chunk in chunks]
    
#     def embed(self):
#         dense_db=FAISS.from_documents(self.documents, self.embedding_model)
#         dense_retriever = dense_db.as_retriever(search_kwargs={"k":5})

#         return dense_retriever
    
#     def ingest(self) -> bool:
#         try:
#             # self.chunk(file_path=file_path)

#             if not self.documents:
#                 raise ValueError("No valid chunks extracted!")
            
#             dense_retriever = self.embed()

#             sparse_retriever = BM25Retriever.from_documents(self.documents)
#             sparse_retriever.k = 5

#             self.retriever=EnsembleRetriever(
#                 retrievers=[sparse_retriever,dense_retriever],
#             )

#             print(self.retriever)

#             return True
        
#         except Exception as e:
#             print(f"Ingestion failed: {e}")
#             self.retriever = None
#             return False

#     def search(self, query:str, top_k:int=10) -> Optional[List[str]]:

#         print(self.retriever)

#         if not self.retriever:
#             print("Retriever not initialized. Ingest documents first.")
#             return None
        
#         try:
#             results = self.retriever.invoke(query)
#             return [(doc.page_content,None) for doc in results[:top_k]]
        
        # except Exception as e:
        #     print(f"Search failed: {str(e)}")
        #     return None
    
#     def build_prompt(self, contexts: List[str], query: str) -> str:
#         """
#         Builds a RAG prompt with clear instructions and context separation.
        
#         Args:
#             contexts: List of context strings retrieved from search
#             query: User's question/query
            
#         Returns:
#             Formatted prompt string
#         """
#         # Join contexts with clear separators
#         context_str = "\n\n--- CONTEXT {} ---\n{}"
#         numbered_contexts = [
#             context_str.format(i+1, ctx.strip())
#             for i, ctx in enumerate(contexts)
#             if ctx.strip()  # Skip empty contexts
#         ]
        
#         prompt_template = (
#             "You are a helpful AI assistant. Answer the question based only on the provided context. "
#             "If the answer is not directly in the context, try to infer it from the information provided'.\n\n"
#             # "Continue your answer after the ':' puncuation"
#             "{contexts}\n\n"
#             "Question: {query}\n"
#             "Answer:"
#         )
        
#         return prompt_template.format(
#             contexts="\n\n".join(numbered_contexts),
#             query=query.strip()
#         )

#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=4, max=10),
#         reraise=True
#     )

#     def send_query(self, query: str, top_k: int = 10) -> Dict[str, Optional[str]]:
#         """
#         Send query to Gemini API with RAG context.
        
#         Args:
#             query: User's question
#             top_k: Number of context passages to retrieve
            
#         Returns:
#             Dictionary containing:
#             - answer: Generated response
#             - contexts: Used context passages
#             - error: Optional error message
#         """
#         try:
#             # Retrieve relevant contexts
#             search_results = self.search(query, top_k=top_k)
#             if not search_results:
#                 return {
#                     "answer": "I couldn't find any relevant information to answer your question.",
#                     "contexts": [],
#                     "error": None
#                 }
                
#             # Extract just the text if search returns (text, score) tuples
#             contexts = [res[0] if isinstance(res, tuple) else res for res in search_results]
            
#             # Build the prompt
#             prompt = self.build_prompt(contexts, query)
            
#             # # Call Gemini API (using hypothetical API structure)
#             # response = requests.post(
#             #     "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
#             #     # "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-latest:generateContent",
#             #     headers={
#             #         "Content-Type": "application/json"
#             #     },
#             #     params={
#             #         "key": self.api_key  # Make sure this is a valid Gemini API key
#             #     },
#             #     json={
#             #         "contents": [
#             #             {
#             #                 "parts": [
#             #                     {"text": prompt}
#             #                 ]
#             #             }
#             #         ],
#             #         "generationConfig": {
#             #             "temperature": 0.7,
#             #             "maxOutputTokens": 1000,
#             #             # "stopSequences": ["\n\n"]
#             #         }
#             #     },
#             #     timeout=10
#             # )

#             content = language_model_api(option=self.option,prompt=prompt)
            
#             return {
#                 # "answer": result.get("text", "").strip(),
#                 "answer":content,
#                 "contexts": contexts,
#                 "error": None
#             }
            
#         except requests.exceptions.RequestException as e:
#             return {
#                 "answer": None,
#                 "contexts": contexts if 'contexts' in locals() else [],
#                 "error": f"API request failed: {str(e)}"
#             }
#         except Exception as e:
#             return {
#                 "answer": None,
#                 "contexts": contexts if 'contexts' in locals() else [],
#                 "error": f"Unexpected error: {str(e)}"
#             }

