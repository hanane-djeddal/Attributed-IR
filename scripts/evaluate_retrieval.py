import os
import sys
import pandas as pd
from tabulate import tabulate

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.evalution.retrieval_metrics import *


def main():
    results_folder = CONFIG["retrieval"]["results_folder"]
    results_file = CONFIG["retrieval"]["results_file"]
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
