from abc import ABC, abstractmethod
from typing import List

class Embedding(ABC):
    @abstractmethod
    def embed(self, chunks: List[str], batch_size:int) -> List[List[float]]:
        '''Input chunks and output embeddings to have semantic meaning'''
        pass