# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import RAG.RAG_Interfaces.Chunker as chunker
# import RAG.RAG_Interfaces.Embedding as embedder
# import RAG.RAG_Interfaces.Query as querier
# import RAG.RAG_Interfaces.Ingester as ingester
# import RAG.RAG_Interfaces.Search as searcher
# import RAG.RAG_Interfaces.Prompter as prompter

from RAG.RAG_Interfaces.Chunker import Chunker
from RAG.RAG_Interfaces.Embedding import Embedding
from RAG.RAG_Interfaces.Query import QueryEngine
from RAG.RAG_Interfaces.Ingester import Ingester
from RAG.RAG_Interfaces.Search import Searcher
from RAG.RAG_Interfaces.Prompter import Prompter

class RAGEngine(Chunker, Embedding, QueryEngine, Ingester, Searcher, Prompter):
    """Main interface that RAG implementations will use"""
    pass