import os
import sys
import pandas as pd
import argparse
import json
from tabulate import tabulate

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.evalution.retrieval_metrics import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    results_folder = (
        CONFIG["retrieval"]["experiment_path"] + CONFIG["retrieval"]["experiment_name"]
    )
    results_file = (
        args.file
        if args.file
        else f"{results_folder}/{CONFIG['retrieval']['results_file']}"
    )
    if results_file.endswith(".json"):
        print("Loading file : ", results_file)
        with open(results_file) as f:
            data = json.load(f)
            results = pd.json_normalize(data)
    elif results_file.endswith(".csv"):
        print("Loading file : ", results_file)
        results = pd.read_csv(
            results_file,
            encoding="latin-1",
            converters={
                CONFIG["column_names"]["passages"]: eval,
                CONFIG["column_names"]["gold_passages"]: eval,
            },
        )
    results["retrieved_ids"] = results[CONFIG["column_names"]["passages"]].apply(
        lambda x: [d["docid"] for d in x]
    )
    results["reference_ids"] = results[CONFIG["column_names"]["gold_passages"]].apply(
        lambda x: [d["docid"] for d in x]
    )

    performance_df = compute_agg_performance(results)
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto"))

    performance_df.to_csv(os.path.join(results_folder, "performance.csv"))


if __name__ == "__main__":
    main()
