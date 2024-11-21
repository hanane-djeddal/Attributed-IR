import os
import sys
import torch
import pandas as pd
import datasets
import argparse
import json

import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.retrieval.retrieve_bm25_monoT5 import Retriever

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=str,
        default="miracl-v1.0-en",
        help="Name of a corpus index from pyserini LuceneSearcher.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="miracl/hagrid",
        help="Valid name of dataset if available on Huggingface, for example : 'miracl/hagrid' ",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="If you provide dataset from huggingface, provide the split name: dev, test, train",
    )

    parser.add_argument(
        "--data_path", type=str, default=None, help="Custom path to dataset file"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--nb_passages",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    ranker = Retriever(index=args.index)

    torch.set_grad_enabled(False)

    results = []

    if args.data_path:
        if CONFIG["data_path"].endswith(".json"):
            print("Loading data : ", CONFIG["data_path"])
            with open(CONFIG["data_path"]) as f:
                dataset = json.load(f)
        elif CONFIG["data_path"].endswith(".csv"):
            print("Loading data : ", CONFIG["data_path"])
            dataframe = pd.read_csv(
                CONFIG["data_path"],
                encoding="latin-1",
            )
            dataset = dataframe.to_dict("records")
    elif args.dataset:
        dataset = datasets.load_dataset(
            args.dataset,
            split=args.split,
        )
    else:
        print(
            "Data file needs to be a json or a csv file. If the dataset is a huggingface downloadable dataset please specify it through the 'dataset' argument"
        )

    for idx, row in enumerate(tqdm(dataset)):
        query = row[CONFIG["column_names"]["query"]]

        top_docs = ranker.search(query, k=args.nb_passages)
        doc_text = [doc["text"] for doc in top_docs]
        row[CONFIG["column_names"]["passages"]] = doc_text
        row["scores"] = [doc["score"] for doc in top_docs]
        results.append(row)
    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    experiment_folder = (
        CONFIG["retrieval"]["experiment_path"] + CONFIG["retrieval"]["experiment_name"]
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        print("New directory for experiment is created: ", experiment_folder)
    results_file = (
        args.results_file
        if args.results_file
        else f"{experiment_folder}/{CONFIG['retrieval']['results_file']}"
    )
    results_df.to_csv(results_file, index=False)
    print("Results saved in", results_file)


if __name__ == "__main__":
    main()
