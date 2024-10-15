import os
import sys
import torch
import pandas as pd
from nltk import sent_tokenize

import re


os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"

import time
from tqdm import tqdm

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG
from src.retrieval.retrieve_bm25_monoT5 import Retriever

Sub_Query = True


def reposition_period(text):
    # Regular expression to find ". [number]"
    pattern = re.compile(r"\.\s*\[\s*(\d+)\s*\]")
    # Replace with "[number] ."
    replaced_text = pattern.sub(r" [\1] .", text)
    return replaced_text


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
    experiment = CONFIG["architectures"]["GTR"]

    print("Indexing corpus with BM25")
    ranker = Retriever(index="miracl-v1.0-en")

    print("Loading queries from MIRACL topics dataset ")

    answer_file = CONFIG["retrieval"]["posthoc_retrieval_file"]
    anwsers_as_queries = pd.read_csv(answer_file, index_col=0)
    #### process generated text
    pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
    anwsers_as_queries["processed_generated_text"] = anwsers_as_queries.apply(
        lambda x: x[CONFIG["column_names"]["prediction"]].replace("<|endoftext|>", ""),
        axis=1,
    )
    anwsers_as_queries["processed_generated_text"] = anwsers_as_queries[
        "processed_generated_text"
    ].str.replace(pattern, "", regex=True)

    results = []

    dataset = anwsers_as_queries  # miracl["dev"], hagrid
    for idx, row in tqdm(dataset.iterrows()):
        if idx == 5:
            break
        question = row["query"]
        answer = row["processed_generated_text"]

        if Sub_Query:

            data = {
                "query": question,
                "generated_answer": answer,
                "gold_truth": row["answers"],
                "gold_quotes": row["quotes"],
                "sub_queries": [],
            }

            for sub_query in sent_tokenize(answer):
                query = " ".join([question, sub_query])
                passages, score = retreive(sub_query, ranker)
                data["sub_queries"].append(
                    {
                        "sub_query": sub_query,
                        "retrieved_passages": passages[
                            : CONFIG["retrieval"]["nb_passages"]
                        ],
                        "score": score,
                    }
                )

        else:
            query = " ".join([question, answer])
            passages, score = retreive(query, ranker)

            data = {
                "query": question,
                "generated_answer": answer,
                "gold_truth": row["answers"],
                "gold_quotes": row["quotes"],
                "retrieved_passages": passages[: CONFIG["retrieval"]["nb_passages"]],
                "score": score,
            }

        results.append(data)

    ## creating attributed anwsers
    answers_with_citation = ""
    all_docs = []
    for _, row in results.iterrows():
        answer = ""
        docs = []
        for q in row["sub_queries"]:
            citedanswer = reposition_period(
                q["sub_query"] + " [" + str(len(docs) + 1) + "] "
            )
            docs.append(q["retrieved_passages"][0])
            answer += citedanswer
        answers_with_citation.append(answer)
        all_docs.append(docs)

    results["output"] = answers_with_citation
    results["docs"] = all_docs

    end = time.time()
    print("time: ", end - start)
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(CONFIG["retrieval"]["results_file_posthoc"])


main()
