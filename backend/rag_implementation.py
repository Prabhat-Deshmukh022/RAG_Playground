import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG.RAG_Engine import RAGEngine
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import faiss
import requests

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class Basic_RAG(RAGEngine):

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # Target size of chunks
            chunk_overlap=50,       # Overlap between chunks
            length_function=len,    # How to measure chunk size
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs, sentences, words
        )

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)

        self.index=None
        self.chunks=[]
        self.dim=None

        self.api_key = GEMINI_API_KEY

    def chunk(self, file_path:str) -> List[str]:
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()

        chunks = self.text_splitter.split_documents(documents)

        return [chunk.page_content for chunk in chunks]
        
    def embed(self, chunks:list[str], batch_size:int = 32) -> np.ndarray:
        self.model.eval()

        all_embeddings=[]

        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Perform mean pooling with attention mask
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Expand mask to match embedding dimensions
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                
                # Sum embeddings along axis 1, but ignore padding tokens
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                
                # Clamp to avoid division by zero
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Get mean embeddings
                mean_embeddings = sum_embeddings / sum_mask
                
                # Normalize embeddings
                normalized_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
                
                all_embeddings.append(normalized_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def ingest(self, file_path:str) -> bool:
        try:
            self.chunks=self.chunk(file_path=file_path)

            if not self.chunks:
                raise ValueError("No valid chunks extracted!")
            
            vectors = self.embed(chunks=self.chunks)
            self.dim = vectors.shape[1]

            faiss.normalize_L2(vectors)

            self.index = faiss.IndexFlatIP(self.dim)

            self.index.add(vectors)

            return True
        
        except Exception as e:
            print(f"Ingestion failed: {str(e)}")
            self.index = None
            self.chunks = []
            return False

    def search(self, query:str, top_k:int = 3) -> Optional[List[str]]:
        if not self.index or not self.chunks:
            print("Index not initialized - please ingest documents first")
            return None
            
        try:
            # Embed and normalize query
            query_vector = self.embed([query])
            faiss.normalize_L2(query_vector)
            
            # Search with score thresholding
            distances, indices = self.index.search(query_vector, top_k)
            
            # Convert FAISS inner product to cosine similarity (0-1)
            scores = [max(0, 1 - distance) for distance in distances[0]]
            
            # Return chunks with their scores
            return [
                (self.chunks[idx], float(score))
                for idx, score in zip(indices[0], scores)
                if idx >= 0  # FAISS returns -1 for invalid indices
            ]
            
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return None

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
            "If you don't know the answer, say 'I don't know'.\n\n"
            # "Continue your answer after the ':' puncuation"
            "{contexts}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        
        return prompt_template.format(
            contexts="\n\n".join(numbered_contexts),
            query=query.strip()
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )

    def send_query(self, query: str, top_k: int = 3) -> Dict[str, Optional[str]]:
        """
        Send query to Gemini API with RAG context.
        
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
            
            # Call Gemini API (using hypothetical API structure)
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                # "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-latest:generateContent",
                headers={
                    "Content-Type": "application/json"
                },
                params={
                    "key": self.api_key  # Make sure this is a valid Gemini API key
                },
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1000,
                        # "stopSequences": ["\n\n"]
                    }
                },
                timeout=10
            )

            response.raise_for_status()  # Raises exception for 4XX/5XX responses
            result = response.json()

            print(response)
            print(result)
            
            return {
                # "answer": result.get("text", "").strip(),
                "answer":result.get("candidates")[0].get("content")['parts'][0]['text'],
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

