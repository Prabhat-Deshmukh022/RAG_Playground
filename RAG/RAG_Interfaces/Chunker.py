from abc import ABC, abstractmethod
from typing import List

class Chunker(ABC):
    @abstractmethod
    def chunk(self, file_path: str) -> List[str]:
        """Split raw text into chunks suitable for retrieval."""
        pass
