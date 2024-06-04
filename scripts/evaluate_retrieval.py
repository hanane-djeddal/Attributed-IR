import os
import sys
import pandas as pd
from tabulate import tabulate

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.evalution.retrieval_metrics import *


def prepare_gold_dataset(file_name: str) -> pd.DataFrame:
    """Aggregates all the relevant passages to a query
    Args:
        file: the path to gold_truth file which has pairs of (query id, passage_id)
    Returns:
        pd.DataFrame: A dataframe with pairs of (query id, list of all relevant passages)
    """
    file = open(file_name)
    lines = file.readlines()
    gold_truth_passages = []
    gold_truth_queries = []

    query_id = lines[0].split()[0]
    passages = []

    for line in lines:
        inputs = line.split()
        new_query_id = inputs[0]
        if query_id != new_query_id:
            gold_truth_passages.append(passages)
            gold_truth_queries.append(query_id)
            passages = []
            query_id = new_query_id
        else:
            passages.append(inputs[2])

    gold_truth = pd.DataFrame.from_dict(
        {"query_id": gold_truth_queries, "gold_passages": gold_truth_passages}
    )
    return gold_truth


def main():
    results_folder = CONFIG["hagrid_miracl"]["results_folder"]
    results_file = CONFIG["hagrid_miracl"]["results_file"]
    print("Loadinf data from file: ", results_file)
    results = pd.read_csv(
        results_file, encoding="latin-1", converters={"retrieved_passages": eval}
    )
    results = pd.read_csv(
        results_file,
        converters={
            "retrieved_ids": eval,
            "retrieved_passages": eval,
            "score": eval,
            "reference_ids": eval,
        },
        index_col=[0],
    )

    performance_df = compute_agg_performance(results)
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto"))

    performance_df.to_csv(os.path.join(results_folder, "performance.csv"))


if __name__ == "__main__":
    main()
