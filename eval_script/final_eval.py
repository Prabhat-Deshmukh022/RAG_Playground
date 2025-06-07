import pandas as pd
from metrics import compute_precision_recall_f1, answer_relevance_score, coverage_score

# Read the evaluation results CSV
df = pd.read_csv("rag_llm_evaluation_results2.csv")

required_cols = {"question", "ground_truth", "LLM_Type", "RAG_Type", "answer", "contexts", "response_time"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Group by LLM_Type and RAG_Type
grouped = df.groupby(["LLM_Type", "RAG_Type"])

for (llm_type, rag_type), group in grouped:
    eval_rows = []
    response_times = []
    for _, row in group.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        answer = row["answer"]
        contexts = row["contexts"]
        response_time = row["response_time"]
        response_times.append(response_time)

        # If ground_truth is a list in string form, convert to list
        if isinstance(ground_truth, str) and ground_truth.startswith("[") and ground_truth.endswith("]"):
            true_answers = [x.strip().strip("'\"") for x in ground_truth[1:-1].split(",")]
        else:
            true_answers = [ground_truth]

        # If contexts is a list in string form, convert to list
        if isinstance(contexts, str) and contexts.startswith("[") and contexts.endswith("]"):
            context_chunks = [x.strip().strip("'\"") for x in contexts[1:-1].split(",")]
        else:
            context_chunks = [contexts]

        prf = compute_precision_recall_f1(true_answers, answer)
        relevance = answer_relevance_score(answer, context_chunks)
        coverage = coverage_score(answer, context_chunks)

        eval_rows.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1_score": prf["f1_score"],
            "relevance": relevance,
            "coverage": coverage,
            "response_time": response_time
        })

    eval_df = pd.DataFrame(eval_rows)
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    # Add avg response time as a separate row at the end
    avg_row = {col: "" for col in eval_df.columns}
    avg_row["question"] = "AVG_RESPONSE_TIME"
    avg_row["response_time"] = round(avg_response_time, 3)
    eval_df = pd.concat([eval_df, pd.DataFrame([avg_row])], ignore_index=True)
    eval_df.to_csv(f"eval_{llm_type}_{rag_type}.csv", index=False)