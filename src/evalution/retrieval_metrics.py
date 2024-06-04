import os
import sys
import pandas as pd
from typing import Tuple

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)


def precision_and_recall_at_k(answers, gold_answers, top_k) -> Tuple[float, float]:
    """Computes precision and recall metrics @K

    Args:
        answers: List of answers given by the retriever
        gold_answers: Answers annotated as relevant
        top_k: Consider top_k to compute metric

    Returns:
        A tuple with precision@k and recall@k
    """
    try:
        unique_answers = set(list(dict.fromkeys(answers))[:top_k])
        gold_answers_k = set(gold_answers)
        matched = unique_answers.intersection(gold_answers_k)

        precision = len(matched) / len(unique_answers)
        recall = len(matched) / len(gold_answers_k)

        return precision, recall
    except (TypeError, ZeroDivisionError):
        return 0, 0


def compute_raw_performance(annotated_dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute performance for a DataFrame containing the answers
       and the annotated data

    Args:
        annotated_dataset (pd.DataFrame): Dataset with answers and annotated data

    Returns:
        pd.DataFrame: A new DataFrame with precision and recall @ 1, 3, 5, 10
    """
    annotated_dataset_with_perf = annotated_dataset.copy()
    for top_k in [1, 3, 5, 10]:
        annotated_dataset_with_perf[[f"P@{top_k}", f"R@{top_k}"]] = (
            annotated_dataset_with_perf.apply(
                lambda x: precision_and_recall_at_k(
                    x["retrieved_ids"], x["reference_ids"], top_k
                ),
                result_type="expand",
                axis=1,
            )
        )

    return annotated_dataset_with_perf


def compute_agg_performance(annotated_dataset: pd.DataFrame) -> pd.DataFrame:
    annotated_dataset_with_perf = compute_raw_performance(annotated_dataset)
    top_k_values = [1, 3, 5, 10]
    agg_dict = {
        **{f"P@{top_k}": "mean" for top_k in top_k_values},
        **{f"R@{top_k}": "mean" for top_k in top_k_values},
    }
    metrics = annotated_dataset_with_perf.apply(agg_dict)
    performance = {
        "Precision": {f"@{top_k}": metrics[f"P@{top_k}"] for top_k in top_k_values},
        "Recall": {f"@{top_k}": metrics[f"R@{top_k}"] for top_k in top_k_values},
    }
    return pd.DataFrame(performance).T
