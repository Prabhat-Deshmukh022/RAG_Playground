from abc import ABC, abstractmethod
from typing import List

class QueryEngine(ABC):
    @abstractmethod
    def send_query(self, prompt:str) -> dict:
        '''Obtains prompt, performs similarity operation and returns a dict with required values'''
        pass