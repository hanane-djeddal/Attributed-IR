import os
import sys
import datasets

import pandas as pd
import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever
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

    dataset = datasets.load_dataset("miracl/hagrid", split="dev")
    queries_dataset = pd.DataFrame(dataset)

    queries_file = CONFIG["retrieval"]["generated_queries_file"]
    print("Loading generated queries from :", queries_file)
    gen_queries = pd.read_csv(
        queries_file, converters={"generated_text": eval}, index_col=0
    )
    gen_queries = gen_queries.rename(
        columns={"query_id": "hagrid_query_id", "generated_text": "generated_queries"}
    )
    queries_dataset = queries_dataset.merge(gen_queries, on="query")

    #### filtering queries
    if CONFIG["retrieval"]["filter_queries"]:
        print("Filtering queries")
        queries_dataset["generated_queries"] = queries_dataset.apply(
            lambda x: query_filter(x["query"], x["generated_queries"]), axis=1
        )
    else:
        queries_dataset["generated_queries"] = queries_dataset[
            "generated_queries"
        ].apply(lambda x: [q[1:] for q in x if q[0] == " "])

    ranker = Retriever(index="miracl-v1.0-en")

    monot5 = MonoT5(device="cuda")

    results = []

    print("Retrieval")
    aggregation = CONFIG["retrieval"]["query_aggregation"]
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
                        "reference_ids": [q["docid"] for q in row["quotes"]],
                        "answers": row["answers"],
                        "gold_quotes": row["quotes"],
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
                "reference_ids": [q["docid"] for q in row["quotes"]],
                "answers": row["answers"],
                "gold_quotes": row["quotes"],
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
    results_df.to_csv(
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}"
    )
    print(
        "Results saved in",
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}",
    )


if __name__ == "__main__":

    main()
