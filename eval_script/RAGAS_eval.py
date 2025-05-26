import os
import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate
from ragas.llms import BaseRagasLLM
from ragas.embeddings import SentenceTransformersEmbedding
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# 1. Setup HuggingFaceHub Langchain LLM
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",  # or any other supported HF model
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# 2. Wrap it for RAGAS
ragas_llm = BaseRagasLLM(llm=llm)
BaseRagasLLM.set_llm(ragas_llm)

# Optional: set embedding model (optional, but helps for context_precision)
embedding = SentenceTransformersEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
SentenceTransformersEmbedding.set_embedding(embedding)

# 3. Load and clean data
df = pd.read_csv("rag_llm_evaluation_results2.csv")
df["contexts"] = df["contexts"].fillna("[]").apply(ast.literal_eval)

# 4. Convert to HF dataset
dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

# 5. Evaluate
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)
