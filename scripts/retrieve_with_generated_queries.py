import os
import sys


import torch
import pandas as pd
import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever
from src.data.miracl_tools import prepare_qrels_data
from src.models.monoT5 import MonoT5
from src.retrieval.query_aggregation import (
    sort_all_scores,
    rerank_against_query,
    vote,
    query_filter,
    simple_retrieval,
    combSum,
)


def main():
    start = time.time()

    print("Loading queries from MIRACL topics dataset ")

    queries_dataset = prepare_qrels_data(
        topics_file=CONFIG["hagrid_miracl"]["topics_file"],
        qrels_file=CONFIG["hagrid_miracl"]["qrels_file"],
    )

    queries_file = CONFIG["hagrid_miracl"]["generated_queries_file"]
    print("Loading generated queries from :", queries_file)
    gen_queries = pd.read_csv(
        queries_file, converters={"generated_text": eval}, index_col=0
    )
    gen_queries = gen_queries.rename(
        columns={"query_id": "hagrid_query_id", "generated_text": "generated_queries"}
    )
    queries_dataset = queries_dataset.merge(gen_queries, on="query")

    #### filtering queries
    if CONFIG["hagrid_miracl"]["filter_queries"]:
        print("Filtering queries")
        queries_dataset["generated_queries"] = queries_dataset.apply(
            lambda x: query_filter(x["query"], x["generated_queries"]), axis=1
        )
    else:
        queries_dataset["generated_queries"] = queries_dataset[
            "generated_queries"
        ].apply(lambda x: [q[1:] for q in x if q[0] == " "])

    print("Indexing corpus with BM25")
    ranker = Retriever(index="miracl-v1.0-en")

    print("loading models : MonoT5")
    torch.set_grad_enabled(False)

    monot5 = MonoT5(device="cuda")

    results = []

    print("Retrieval")
    aggregation = CONFIG["hagrid_miracl"]["query_aggregation"]
    dataset = queries_dataset
    for _, row in tqdm(dataset.iterrows()):
        query_id = row["query_id"]
        query = row["query"]
        if aggregation == "sort":
            ids, text, score = sort_all_scores(query, row["generated_queries"], ranker)
        elif aggregation == "rerank":
            ids, text, score = rerank_against_query(
                query, row["generated_queries"], ranker, monot5
            )
        elif aggregation == "combSum":
            ids, text, score = combSum(
                query, row["generated_queries"], ranker, MNZ=False
            )
        elif aggregation == "combMNZ":
            ids, text, score = combSum(
                query, row["generated_queries"], ranker, MNZ=True
            )
        elif aggregation == "vote":
            ids, text, score = vote(query, row["generated_queries"], ranker)
        elif aggregation == "summed_vote":
            ids, text, score = vote(
                query,
                row["generated_queries"],
                ranker,
                score_compute="sum",
            )
        elif aggregation == "mean_vote":
            ids, text, score = vote(
                query,
                row["generated_queries"],
                ranker,
                score_compute="mean",
            )
        elif aggregation == "seperate_queries":
            for q in row["generated_queries"]:
                ids, text, score = simple_retrieval(q, ranker)
                results.append(
                    {
                        "query": query,
                        "query_id": query_id,
                        "sub_query": q,
                        "retrieved_ids": ids,
                        "retrieved_passages": [],
                        "score": score,
                        "reference_ids": row["relevant_docids"],
                    }
                )
                ids, text, score = simple_retrieval(query, ranker)

        results.append(
            {
                "query": query,
                "query_id": query_id,
                "sub_query": "",
                "retrieved_ids": ids,
                "retrieved_passages": text,
                "score": score,
                "reference_ids": row["relevant_docids"],
            }
        )
    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(CONFIG["hagrid_miracl"]["results_file"])
    print(
        "Result file:",
        CONFIG["hagrid_miracl"]["results_file"],
    )


if __name__ == "__main__":

    main()