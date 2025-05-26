import langchain
from langsmith import Client
import csv
import os
from dotenv import load_dotenv
from langsmith import wrappers
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
model="mistral-medium"

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def iterate_csv_to_dataset(csv_file_path):
    example=[]
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:

        reader=csv.DictReader(csvfile)
        for row in reader:
            question=row.get("question")
            answer=row.get("answer")
            example.append({
                "inputs":{"question":question},
                "outputs":{"answer":answer}
            })
    
    return example

# example = iterate_csv_to_dataset("ground_truth.csv")

langsmith_client = Client()

def create_dataset(example):
    dataset_name="New_Dataset"
    dataset = langsmith_client.create_dataset(dataset_name=dataset_name)
    langsmith_client.create_examples(
        dataset_id=dataset.id,
        examples=example
    )
    return dataset_name

def make_examples(predicted_file, ground_truth_file):
    ground_truth_map = {}
    
    # Read ground truth into a map for quick lookup
    with open(ground_truth_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth_map[row["question"]] = row["answer"]
    
    examples = []
    with open(predicted_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"]
            model_answer = row["answer"]
            gt_answer = row["Reference_A"]

            examples.append({
                "inputs": {"question": question},
                "outputs": {"response": model_answer},
                "reference_outputs": {"answer": gt_answer}
            })
    
    return examples

example = make_examples("rag_llm_evaluation_results.csv","ground_truth.csv")
dataset_name=create_dataset(example=example)

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""You are grading the following question:
{inputs['question']}
Here is the real answer:
{reference_outputs.get('answer', '')}
You are grading the following predicted answer:
{outputs.get('response', '')}
Respond with CORRECT or INCORRECT:
Grade:
"""
    response = mistral_client.chat.complete(
        model="mistral-large-2402",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content.strip().upper()
    
    return response == "CORRECT"

def concision(outputs: dict, reference_outputs: dict) -> bool:
    response = outputs.get("response", "")
    reference = reference_outputs.get("answer", "")
    return int(len(response) < 2 * len(reference))

default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."

def my_app(question: str, model: str = "mistral-large-2402", instructions: str = default_instructions) -> str:
    return mistral_client.chat.complete(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content

def ls_target(inputs: dict) -> dict:
    return {"outputs": {"response": my_app(inputs["question"])}}

experiment_results = langsmith_client.evaluate(
    ls_target, # Your AI system
    data=dataset_name, # The data to predict and grade over
    evaluators=[concision, correctness], # The evaluators to score the results
    experiment_prefix="RAG_VALIDATE", # A prefix for your experiment names to easily identify them
)