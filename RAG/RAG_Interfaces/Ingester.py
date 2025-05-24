from abc import ABC, abstractmethod

class Ingester(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> bool:
        '''For building FAISS index'''
        pass