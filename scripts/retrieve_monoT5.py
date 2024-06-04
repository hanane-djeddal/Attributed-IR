import os
import sys
import torch
import pandas as pd


import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.data.miracl_tools import prepare_qrels_data

from src.retrieval.retrieve_bm25_monoT5 import Retriever


def main():
    start = time.time()

    print("Indexing corpus with BM25")
    ranker = Retriever(index="miracl-v1.0-en")

    print("Loading queries from MIRACL topics dataset ")

    queries_dataset = prepare_qrels_data(
        topics_file=CONFIG["hagrid_miracl"]["topics_file"],
        qrels_file=CONFIG["hagrid_miracl"]["qrels_file"],
    )

    print("loading models : MonoT5")
    torch.set_grad_enabled(False)

    results = []

    dataset = queries_dataset
    for _, row in tqdm(dataset.iterrows()):
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
                "relevant_ids": row["relevant_docids"],
            }
        )
    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(CONFIG["hagrid_miracl"]["results_file"])


if __name__ == "__main__":
    main()
