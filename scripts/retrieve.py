import os
import sys
import torch
import pandas as pd
import datasets

import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.retrieval.retrieve_bm25_monoT5 import Retriever


def main():
    start = time.time()

    ranker = Retriever(index="miracl-v1.0-en")

    torch.set_grad_enabled(False)

    results = []

    dataset = datasets.load_dataset("miracl/hagrid", split="dev")
    for row in tqdm(dataset):
        query_id = row["query_id"]
        query = row["query"]

        top_docs = ranker.search(query)
        docids = [doc["id"] for doc in top_docs]
        doc_text = [doc["text"] for doc in top_docs]
        results.append(
            {
                "query": query,
                "query_id": query_id,
                "retrieved_ids": docids,
                "retrieved_passages": doc_text,
                "reference_ids": [q["docid"] for q in row["quotes"]],
                "gold_quotes": row["quotes"],
                "answers": row["answers"],
            }
        )
    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    experiment_folder = (
        CONFIG["retrieval"]["experiment_path"] + CONFIG["retrieval"]["experiment_name"]
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        print("New directory for experiment is created: ", experiment_folder)
    results_df.to_csv(f"{experiment_folder}/{CONFIG['retrieval']['results_file']}")
    print(
        "Results saved in", f"{experiment_folder}/{CONFIG['retrieval']['results_file']}"
    )


if __name__ == "__main__":
    main()
