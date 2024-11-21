import os
import sys
import datasets
import argparse

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


def format_support_passages(passages_text, passages_ids):
    passages = []
    for i in range(len(passages_text)):
        passages.append(
            {"idx": i + 1, "docid": passages_ids[i], "text": passages_text[i]["text"]}
        )
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=None, help="File containing sunqueries"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--index",
        type=str,
        default="miracl-v1.0-en",
        help="Name of a corpus index from pyserini LuceneSearcher.",
    )

    args = parser.parse_args()
    start = time.time()
    query_gen_folder = (
        CONFIG["query_generation"]["experiment_path"]
        + CONFIG["query_generation"]["experiment_name"]
    )
    queries_file = (
        args.input_file
        if args.input_file
        else f"{query_gen_folder}/{CONFIG['query_generation']['results_file']}"
    )
    print("Loading generated queries from :", queries_file)
    queries_dataset = pd.read_csv(queries_file, converters={"generated_queries": eval})

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

    ranker = Retriever(index=args.index)

    monot5 = MonoT5(device="cuda")

    results = []

    print("Retrieval")
    aggregation = CONFIG["retrieval"]["query_aggregation"]
    dataset = queries_dataset
    nb_passages = CONFIG["retrieval"]["nb_passages"]
    for idx, row in tqdm(dataset.iterrows()):
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

                row["sub_query"] = q
                row[CONFIG["column_names"]["passages"]] = format_support_passages(
                    text, ids
                )
                row["scores"] = score
                results.append(row)
            ids, text, score = simple_retrieval(query, ranker)
        row["sub_query"] = ""
        row[CONFIG["column_names"]["passages"]] = format_support_passages(text, ids)[
            :nb_passages
        ]
        row["scores"] = score[:nb_passages]
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
    results_df.to_csv(
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}",
        index=False,
    )
    print(
        "Results saved in",
        f"{experiment_folder}/{CONFIG['retrieval']['query_gen_results_file']}",
    )


if __name__ == "__main__":

    main()
