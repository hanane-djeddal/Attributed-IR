import os
import sys
import torch
import pandas as pd
from nltk import sent_tokenize


import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever

Sub_Query = True


def format_support_passages(passages_text, passages_ids):
    passages = []
    for i in range(len(passages_text)):
        passages.append(
            {"idx": i + 1, "docid": passages_ids[i], "text": passages_text[i]["text"]}
        )
    return passages


def retreive(query, ranker):

    top_docs = ranker.search(query)
    ids = [doc["id"] for doc in top_docs]
    text = [doc["text"] for doc in top_docs]
    score = [doc["score"] for doc in top_docs]

    passages = format_support_passages(text, ids)

    return passages, score


def main():
    start = time.time()

    print("Indexing corpus with BM25")
    ranker = Retriever(index="miracl-v1.0-en")

    print("Loading queries from MIRACL topics dataset ")

    answer_file = CONFIG["retrieval"]["posthoc_retrieval_file"]
    anwsers_as_queries = pd.read_csv(answer_file, index_col=0)
    #### process generated text
    pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
    anwsers_as_queries["processed_generated_text"] = anwsers_as_queries.apply(
        lambda x: x["generated_text"].replace("<|endoftext|>", ""), axis=1
    )
    anwsers_as_queries["processed_generated_text"] = anwsers_as_queries[
        "processed_generated_text"
    ].str.replace(pattern, "", regex=True)

    results = []

    dataset = anwsers_as_queries  # miracl["dev"], hagrid
    for _, row in tqdm(dataset.iterrows()):
        question = row["query"]
        answer = row["processed_generated_text"]

        if Sub_Query:

            data = {
                "query": question,
                "generated_answer": answer,
                "gold_truth": row["gold_truth"],
                "gold_quotes": row["gold_quotes"],
                "sub_queries": [],
            }

            for sub_query in sent_tokenize(answer):
                query = " ".join([question, sub_query])
                passages, score = retreive(sub_query, ranker)
                data["sub_queries"].append(
                    {
                        "sub_query": sub_query,
                        "retrieved_passages": passages,
                        "score": score,
                    }
                )

        else:
            query = " ".join([question, answer])
            passages, score = retreive(query, ranker)

            data = {
                "query": question,
                "generated_answer": answer,
                "gold_truth": row["gold_truth"],
                "gold_quotes": row["gold_quotes"],
                "retrieved_passages": passages,
                "score": score,
            }

        results.append(data)

    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(CONFIG["retrieval"]["results_file"])


main()
