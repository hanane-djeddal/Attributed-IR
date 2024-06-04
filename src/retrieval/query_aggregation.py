import os
import sys
from typing import List
import spacy
import torch.nn as nn
import torch
from pyserini.search.lucene import LuceneSearcher

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)


PRON = [" he ", " she ", " his ", " hers "]


def is_ambiguous(sentence):
    nlp = spacy.load("en_core_web_md")

    doc = nlp(sentence)

    entities = [ent.text for ent in doc.ents]
    if "which century" in entities:
        entities.remove("which century")

    if len(entities):
        return False  # sentence is clear

    return True  # sentence is ambiguous


def query_filter(original_query: str, generated_queries: List[str]) -> List[str]:
    if original_query in generated_queries:
        generated_queries.remove(original_query)

    for query in generated_queries:
        if any(prn in query for prn in PRON):
            if is_ambiguous(query):
                generated_queries.remove(query)
    return generated_queries


def deduplicate(doc_list):
    set_ids = []
    set_candidates = []
    for p in doc_list:
        if p["id"] not in set_ids:
            set_candidates.append(p)
            set_ids.append(p["id"])
    return set_candidates


def simple_retrieval(original_query: str, ranker):
    """
    For each query (original + generated), the function retrives documents using (BM25 + MonoT5), the documents are then sorted byt their MonotT5 score (qi,d)
    """

    top_docs = ranker.search(original_query)
    ids = [doc["id"] for doc in top_docs]
    text = [doc["text"] for doc in top_docs]
    score = [doc["score"] for doc in top_docs]

    return ids, text, score


def sort_all_scores(original_query: str, generated_queries: List[str], ranker):
    """
    For each query (original + generated), the function retrives documents using (BM25 + MonoT5), the documents are then sorted byt their MonotT5 score (qi,d)
    """
    scores_by_id = {}
    text_by_ids = {}
    queries = [original_query] + generated_queries
    for q in queries:
        top_docs = ranker.search(q)
        for doc in top_docs:
            if doc["id"] in list(scores_by_id.keys()):
                if scores_by_id[doc["id"]] < doc["score"]:
                    scores_by_id[doc["id"]] = doc["score"]
            else:
                scores_by_id[doc["id"]] = doc["score"]
                text_by_ids[doc["id"]] = doc["text"]

        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(), key=lambda item: item[1], reverse=True
            )
        }
        ids = list(selected_docs.keys())
        text = [text_by_ids[i] for i in ids]
        score = list(selected_docs.values())
    return ids, text, score


def rerank_against_query(
    original_query: str, generated_queries: List[str], ranker, monot5
):
    all_candidates = []
    queries = [original_query] + generated_queries
    for q in queries:
        top_docs = ranker.search(q)

        all_candidates.extend(top_docs[:20])

    set_candidates = deduplicate(all_candidates)
    reranked_all_docs = monot5.rerank(original_query, set_candidates)

    ids = [doc["id"] for doc in reranked_all_docs]
    text = [doc["text"] for doc in reranked_all_docs]
    score = [doc["score"] for doc in reranked_all_docs]
    return ids, text, score


def vote(
    original_query: str,
    generated_queries: List[str],
    ranker,
    score_compute="count",  # sum, mean, weighed sum
):
    """
    For each query (original + generated), the function retrives documents using (BM25 + MonoT5), the documents are then sorted byt their MonotT5 score (qi,d)
    """
    scores_by_id = {}
    text_by_ids = {}
    queries = [original_query] + generated_queries
    for q in queries:
        top_docs = ranker.search(q)

        top_20 = top_docs[:20]
        for doc in top_20:
            if doc["id"] in list(scores_by_id.keys()):
                scores_by_id[doc["id"]].append(doc["score"])
            else:
                scores_by_id[doc["id"]] = [doc["score"]]
                text_by_ids[doc["id"]] = doc["text"]
    if score_compute == "count":  # sort by the number of times it was retrieved
        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(), key=lambda item: len(item[1]), reverse=True
            )
        }
        score = [sum(s) for s in list(selected_docs.values())]
    elif score_compute == "sum":
        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(), key=lambda item: sum(item[1]), reverse=True
            )
        }
        score = [sum(s) for s in list(selected_docs.values())]
    elif score_compute == "mean":
        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(),
                key=lambda item: sum(item[1]) / len(item[1]),
                reverse=True,
            )
        }
        score = [sum(s) / len(s) for s in list(selected_docs.values())]
    ids = list(selected_docs.keys())
    text = [text_by_ids[i] for i in ids]
    return ids, text, score


def rerank_original_pool_postion_vote(
    original_query: str, generated_queries: List[str], ranker
):
    """ """

    queries = [original_query] + generated_queries
    doc_by_position = [{} for i in range(20)]
    for q in queries:
        top_docs = ranker.search(q)
        for i in range(20):
            if top_docs[i]["id"] in doc_by_position[i]:
                doc_by_position[i][top_docs[i]["id"]].append(top_docs[i]["score"])
            else:
                doc_by_position[i][top_docs[i]["id"]] = [top_docs[i]["score"]]

    ordered_doc_by_position = [
        {
            k: v
            for k, v in sorted(
                doc_by_position[i].items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        }
        for i in doc_by_position.keys()
    ]
    selected_docs = {
        list(d.keys())[0]: sum(d[list(d.keys())[0]]) for d in ordered_doc_by_position
    }
    ids = list(selected_docs.keys())
    text = [top_docs[i] for i in ids]
    score = list(selected_docs.values())
    return ids, text, score


def combSum(
    original_query: str,
    generated_queries: List[str],
    ranker,
    MNZ=False,  # sum, mean, weighed sum
):
    """
    For each query (original + generated), the function retrives documents using (BM25 + MonoT5), the documents are then sorted byt their MonotT5 score (qi,d)
    """
    scores_by_id = {}
    text_by_ids = {}
    queries = [original_query] + generated_queries
    for q in queries:
        top_docs = ranker.search(q)
        scores_list = [d["score"] for d in top_docs]
        max_val = max(scores_list)
        min_val = min(scores_list)

        range_val = max_val - min_val
        for doc in top_docs:
            if doc["id"] in list(scores_by_id.keys()):
                scores_by_id[doc["id"]].append((doc["score"] - min_val) / range_val)
            else:
                scores_by_id[doc["id"]] = [(doc["score"] - min_val) / range_val]
                text_by_ids[doc["id"]] = doc["text"]

    if MNZ == True:
        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(),
                key=lambda item: len(item[1]) * sum(item[1]),
                reverse=True,
            )
        }
        score = [sum(s) * len(s) for s in list(selected_docs.values())]
    else:
        selected_docs = {
            k: v
            for k, v in sorted(
                scores_by_id.items(),
                key=lambda item: sum(item[1]),
                reverse=True,
            )
        }
        score = [sum(s) for s in list(selected_docs.values())]
    ids = list(selected_docs.keys())
    text = [text_by_ids[i] for i in ids]
    return ids, text, score


############ aggregation after retrieval


def PM2_post(
    score_per_query,
    docids_per_query,
):
    score_per_query = score_per_query[-1:] + score_per_query[:-1]
    docids_per_query = docids_per_query[-1:] + docids_per_query[:-1]
    scores_by_id = []
    h = 0.1
    for j in range(len(score_per_query)):
        scores_list = [d for d in score_per_query[j]]

        softmax = nn.Softmax(dim=0)
        proba = softmax(torch.tensor(scores_list, dtype=float))
        top_20 = []
        for i in range(len(score_per_query[j])):
            top_20.append({"id": docids_per_query[j][i], "score": proba[i].item()})
        scores_by_id.append(top_20)
    v = [0.38, 0.24, 0.16, 0.10, 0.12]
    seats = []
    s = [0 for i in range(len(score_per_query))]
    quotient = [0 for i in range(len(score_per_query))]

    selected_docs = []
    for pos in range(20):  # fill 20 seats
        for q in range(len(score_per_query)):  # select an aspect (a sub query)
            if q >= len(v):
                quotient[q] = 0
            else:
                quotient[q] = v[q] / (2 * s[q] + 1)
        max_quotient = 0
        argi = 0
        for quot in range(len(quotient)):
            if quotient[quot] >= max_quotient and len(scores_by_id[argi]) != 0:
                max_quotient = quotient[quot]
                argi = quot

        ### find which document to select from the list of the aspect (subquery)
        max_score = 0
        argd = ""
        max_proba_to_other_queries = []
        for d in scores_by_id[argi]:  # for all documents in the list of apsct argi
            if d["id"] not in selected_docs:  # if document is not already selected
                quotien_to_other_queries = []
                proba_to_other_queries = []
                # find the document that maximizes a score in regard to all other aspects
                for i in range(
                    len(quotient)
                ):  # for all other aspects different than the selected aspect
                    if i != argi:
                        found = False
                        for doc in scores_by_id[
                            i
                        ]:  # if the the document d is in the list of documents returned by aspect i
                            if doc["id"] == d["id"]:
                                proba_to_other_queries.append(doc["score"])
                                quotien_to_other_queries.append(
                                    quotient[i] * doc["score"]
                                )
                                found = True
                        if not found:
                            quotien_to_other_queries.append(0)
                            proba_to_other_queries.append(0)
                    else:
                        proba_to_other_queries.append(d["score"])

                selection_score = h * quotient[argi] * d["score"] + (1 - h) * sum(
                    quotien_to_other_queries
                )
                if selection_score > max_score:  # update max document
                    max_score = selection_score
                    argd = d
                    max_proba_to_other_queries = proba_to_other_queries
        # select the doc
        if argd != "":
            selected_docs.append(argd["id"])
            seats.append(argd)
            for q in range(len(score_per_query)):
                s[q] = s[q] + (
                    max_proba_to_other_queries[q] / sum(max_proba_to_other_queries)
                )
            scores_by_id[argi].remove(argd)

    scores = [d["score"] for d in seats]
    ids = [d["id"] for d in seats]
    return ids, scores
