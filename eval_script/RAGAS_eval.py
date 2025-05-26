import os
import pandas as pd
from datasets import Dataset
from ast import literal_eval
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    AnswerCorrectness,
)

# Set Groq API key
os.environ["GROQ_API_KEY_2"] = os.getenv("GROQ_API_KEY_2")

# Use Groq LLM via Langchain
groq_llm = ChatGroq(model="llama3-70b-8192")
wrapped_llm = LangchainLLMWrapper(langchain_llm=groq_llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load CSV
df = pd.read_csv("rag_llm_evaluation_results2.csv")
df=df.sample(10,random_state=42)

# Convert context strings to list
df["contexts"] = df["contexts"].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Evaluate RAG with RAGAS and Groq
results = evaluate(
    dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        ContextRelevance(),
        AnswerCorrectness(),
    ],
    llm=wrapped_llm,
    embeddings=embeddings
)

print("\n📊 RAGAS Evaluation Results:")
print(results)