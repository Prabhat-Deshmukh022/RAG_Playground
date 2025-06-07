from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List
from difflib import SequenceMatcher


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def compute_precision_recall_f1(true_answers: List[str], generated_answer: str, threshold: float = 0.7):
    matched = [1 if string_similarity(generated_answer, true) >= threshold else 0 for true in true_answers]
    y_true = [1] * len(true_answers)
    precision = precision_score(y_true, matched, zero_division=0)
    recall = recall_score(y_true, matched, zero_division=0)
    f1 = f1_score(y_true, matched, zero_division=0)
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }


def answer_relevance_score(answer: str, chunks: List[str]) -> float:
    scores = [string_similarity(answer, chunk) for chunk in chunks]
    return round(max(scores), 3)


def coverage_score(answer: str, chunks: List[str]) -> float:
    total = len(answer.split())
    covered = sum(1 for word in answer.split() if any(word in chunk for chunk in chunks))
    return round(covered / total, 3) if total > 0 else 0.0