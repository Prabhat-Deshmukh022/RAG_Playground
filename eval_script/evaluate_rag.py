import pandas as pd
import requests
from tqdm import tqdm
import time

API_UPLOAD = "http://localhost:8000/upload"
API_QUERY = "http://localhost:8000/query"

RAG_TYPES = {
    1: "Basic",
    2: "Hybrid",
    3: "AutoMerge"
}

LLM_TYPES = {
    1: "Gemini",
    2: "Mistral",
    3: "Groq"
}

# Use two dummy PDF files for upload
PDF_PATHS = ["AI_PM_2.pdf", "IJSRA-2023-0710.pdf"]

def send_upload(rag_option, llm_option):
    """Send upload request to simulate user config selection with multiple PDFs"""
    files = [
        ('files', open(PDF_PATHS[0], 'rb')),
        ('files', open(PDF_PATHS[1], 'rb'))
    ]
    data = {
        "rag_option": str(rag_option),
        "llm_option": str(llm_option)
    }
    try:
        response = requests.post(API_UPLOAD, files=files, data=data)
        return response.ok
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return False
    finally:
        # Close the file handles
        for _, f in files:
            f.close()

# def query(question):
#     """Send query request and return model-generated answer"""
#     try:
#         response = requests.post(API_QUERY, json={"query": question})
#         if response.ok:
#             return response.json().get("response", "")
#         else:
#             return f"[ERROR] {response.status_code}: {response.text}"
#     except Exception as e:
#         return f"[ERROR] {e}"

def query(question, rag_option):
    """Send query request and return model-generated answer"""
    try:
        # Build URL with option as query parameter
        url = f"{API_QUERY}?option={rag_option}"

        # Send 'question' as form data
        start=time.time()
        response = requests.post(url, data={"query": question})
        end=time.time()

        if response.ok:
            response_time=end-start
            return response.json(),response_time
        else:
            return f"[ERROR] {response.status_code}: {response.text}"
    except Exception as e:
        return f"[ERROR] {e}"


def evaluate(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for rag_option in range(1, 4):
        for llm_option in range(1, 4):
            print(f"\nEvaluating RAG: {RAG_TYPES[rag_option]}, LLM: {LLM_TYPES[llm_option]}")
            if not send_upload(rag_option, llm_option):
                print(f"Failed to upload for RAG={rag_option}, LLM={llm_option}. Skipping...")
                continue

            time.sleep(1)  # Allow backend to register the config

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying"):
                question = row["question"]
                ground_truth = row["answer"]

                result = query(question, rag_option=rag_option)
                if isinstance(result, tuple):
                    result_dict, response_time = result
                else:
                    result_dict, response_time = result, None

                if isinstance(result_dict, dict):
                    model_answer = result_dict.get("answer", "")
                    contexts = result_dict.get("contexts", "")
                else:
                    model_answer = result_dict  # error string
                    contexts = ""

                results.append({
                    "question": question,
                    "answer": model_answer,
                    "contexts":contexts,
                    "ground_truth": ground_truth,
                    "LLM_Type": LLM_TYPES[llm_option],
                    "RAG_Type": RAG_TYPES[rag_option],
                    "response_time":response_time
                })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ Evaluation saved to {output_csv}")

if __name__ == "__main__":
    evaluate("ground_truth.csv", "rag_llm_evaluation_results2.csv")
