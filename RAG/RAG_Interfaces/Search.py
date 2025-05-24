from abc import ABC, abstractmethod
from typing import List

class Searcher(ABC):
    @abstractmethod
    def search(self, query:str, top_k:int) -> List:
        '''Returns top three context vectors'''
        pass