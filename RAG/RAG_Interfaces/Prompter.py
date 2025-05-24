from abc import ABC, abstractmethod
from typing import List

class Prompter(ABC):
    @abstractmethod
    def build_prompt(self, query:str, context:List[str]) -> str:
        '''Building prompt'''
        pass