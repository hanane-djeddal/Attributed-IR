import pandas as pd
import os
import sys
import datasets

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from config import CONFIG


def prepare_qrels_data(topics_file: str, qrels_file: str):
    print("Preparing qrels dataset from manually downloaded files")
    ### columns name (files without headers)
    topic_column_names = ["qid", "query"]
    qrels_column_names = ["qid", "Q0", "docid", "relevance"]
    qrels = pd.read_csv(qrels_file, sep="\t", header=None, names=qrels_column_names)
    topics = pd.read_csv(topics_file, sep="\t", header=None, names=topic_column_names)

    ### only keep relevant documents & aggregate dataset
    relevant_qrels_only = qrels[qrels["relevance"] == 1]
    aggregated_qrels = (
        relevant_qrels_only.groupby("qid")["docid"]
        .apply(list)
        .reset_index(name="relevant_docids")
    )

    ### merge
    df = topics.merge(aggregated_qrels, on="qid")
    df.rename(columns={"qid": "query_id"}, inplace=True)

    assert df["query_id"].nunique() == len(df)
    return df


def test():
    topics_file = CONFIG["hagrid_miracl"]["topics_file"]
    qrels_file = CONFIG["hagrid_miracl"]["qrels_file"]

    df = prepare_qrels_data(topics_file, qrels_file)
    print(df.head(3))
    df["number_of_relevant_docids"] = df["relevant_docids"].apply(lambda x: len(x))
    print(df.head(3))
    print(df["number_of_relevant_docids"].describe())


def compare_to_hagrid():
    print("Loading topics dataset (hargrid queries) from HF : ")
    hagrid = datasets.load_dataset(
        "miracl/hagrid", split="dev", cache_dir=CONFIG["hagrid_miracl"]["cache_dir"]
    )
    topics_file = CONFIG["hagrid_miracl"]["topics_file"]
    qrels_file = CONFIG["hagrid_miracl"]["qrels_file"]
    df = prepare_qrels_data(topics_file, qrels_file)
    print("MIRACL dev size: ", len(df))
    print("HAGRID dev size: ", len(hagrid))
    topics = list(df["query"].unique())
    # print("topics:", topics)
    not_found = 0
    found = 0
    for row in hagrid:
        if row["query"] not in topics:
            # print("Topic from Hagrid not in Miracl!")
            # print(row["query_id"])
            # print("------")
            not_found += 1
        else:
            found += 1
    print("Number of Not found: ", not_found)
    print("Number of found: ", found)


# test()
# compare_to_hagrid()
