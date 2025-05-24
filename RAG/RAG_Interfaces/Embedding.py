from abc import ABC, abstractmethod
from typing import Any

class Embedding(ABC):
    @abstractmethod
    def embed(self, *args, **kwargs) -> Any:
        '''Input chunks and output embeddings to have semantic meaning'''
        pass