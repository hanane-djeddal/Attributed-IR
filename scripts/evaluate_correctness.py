import os
import sys
import numpy as np
import pandas as pd
import argparse
from tabulate import tabulate
import json


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.evalution.generation_metrics import *
from src.data.hagrid_dataset_tools import get_attributable_answer, get_all_answers

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        type=str,
        default="G",
        choices=["G", "RTG-gold", "RTG-vanilla", "RTG-query-gen"],
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--multiple_gold_answers",
        action="store_true",
    )
    parser.add_argument(
        "--per_example_scores",
        action="store_true",
    )
    args = parser.parse_args()
    experiment = CONFIG["architectures"][args.architecture]

    results_folder = experiment["experiment_path"] + experiment["experiment_name"]

    results_file = (
        args.results_file
        if args.results_file
        else (results_folder + "/" + experiment["results_file"])
    )
    multiple_answers = (
        args.multiple_gold_answers
        if args.multiple_gold_answers
        else CONFIG["multiple_gold_answers"]
    )

    print("Loading results file:", results_file)
    generated_column = CONFIG["column_names"]["prediction"]
    reference_column = CONFIG["column_names"]["reference"]
    if results_file.endswith(".json"):
        with open(results_file) as f:
            json_dict = json.load(f)
            results = pd.json_normalize(json_dict["data"])
    elif multiple_answers:
        results = pd.read_csv(results_file, converters={reference_column: eval})
    else:
        results = pd.read_csv(results_file)

    ## processing the generated text to remove system prompt

    pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
    results["processed_generated_text"] = results.apply(
        lambda x: x[generated_column].replace("<|endoftext|>", ""), axis=1
    )
    results["processed_generated_text"] = results[
        "processed_generated_text"
    ].str.replace(pattern, "", regex=True)
    print("example processed text : ", results["processed_generated_text"][0], "\n")

    if CONFIG["dataset"] == "HAGRID":
        results["gold_answer"] = results[reference_column].apply(
            get_attributable_answer
        )
        results = results[results["gold_answer"].str.len() > 0]
    elif multiple_answers:
        results["gold_answer"] = results[reference_column].apply(
            lambda x: x[0][CONFIG["column_names"]["multiple_answers"]]
        )
    else:
        results["gold_answer"] = results[reference_column]

    ### remove citiations from answer
    citation_pattern = r"\[\d+(?:,\s*\d+)*\]"
    results["gold_answer"] = results["gold_answer"].str.replace(
        citation_pattern, "", regex=True
    )
    if multiple_answers:
        # add all answer without citiations
        results["all_gold_answer"] = results[reference_column].apply(
            lambda x: get_all_answers(
                x, answer_kw=CONFIG["column_names"]["multiple_answers"]
            )
        )
        print(
            "example all gold answer without citations: ",
            results["all_gold_answer"][0],
            "\n",
        )

    print("example gold answer without citations: ", results["gold_answer"][0], "\n")

    if experiment["citation"]:
        results["processed_generated_text"] = results[
            "processed_generated_text"
        ].str.replace(citation_pattern, "", regex=True)
        print(
            "example generated answer without citations : ",
            results["processed_generated_text"][0],
            "\n",
        )

    reference_column = "gold_answer"
    generated_column = "processed_generated_text"

    # with only one reference answer
    print("loading rouge metric")
    rouge_scores = rouge(
        results[generated_column],
        results[reference_column],
        CONFIG,
        use_aggregator=False,
        rouge_types=["rougeLsum"],
    )  # new Huggingface version of rouge which returns one aggregated value
    print("loading rouge metric dataset version ")
    rouge_scores_ov = rouge_detailed_ov(
        results[generated_column], results[reference_column], CONFIG
    )  # Huggingface old version of rouge that details precision/recall/fscore
    print("rouge score:", rouge_scores)
    print("loading bert metric")
    bert_scores = bert_score(
        results[generated_column], results[reference_column], CONFIG
    )
    # bleu score
    print("load bleu score")
    bleu_score = bleu(results[generated_column], results[reference_column], CONFIG)
    rouge_scores_ov_all = None
    bert_scores_all = None
    if multiple_answers:
        # with all reference answer

        reference_column = "all_gold_answer"

        print("loading rouge metric dataset version, all")
        rouge_scores_ov_all = rouge_detailed_ov_all(
            results[generated_column], results[reference_column], CONFIG
        )

        print("loading bert metric, all")
        bert_scores_all = bert_metric_all(
            results[generated_column], results[reference_column], CONFIG
        )

    performance = {
        "Rouge": {
            "Precision": round(rouge_scores_ov["rougeLsum"][1][0] * 100, 2),
            "Recall": round(rouge_scores_ov["rougeLsum"][1][1] * 100, 2),
            "fmeasure": round(rouge_scores_ov["rougeLsum"][1][2] * 100, 2),
        },
        "Bert": {
            "Precision": round(np.array(bert_scores["precision"]).mean() * 100, 2),
            "Recall": round(np.array(bert_scores["recall"]).mean() * 100, 2),
            "fmeasure": round(np.array(bert_scores["f1"]).mean() * 100, 2),
        },
        "Aggregated Rouge": {
            "Precision": round(rouge_scores["rougeLsum"] * 100, 2),
            "Recall": round(rouge_scores["rougeLsum"] * 100, 2),
            "fmeasure": round(rouge_scores["rougeLsum"] * 100, 2),
        }
    }
    if args.multiple_gold_answers:
        performance_all = {
            "Rouge All": {
                "Precision": round(np.mean(rouge_scores_ov_all["precision"]) * 100, 2),
                "Recall": round(np.mean(rouge_scores_ov_all["recall"]) * 100, 2),
                "fmeasure": round(np.mean(rouge_scores_ov_all["fmeasure"]) * 100, 2),
            },
            "Bert All": {
                "Precision": round(np.mean(bert_scores_all["precision"]) * 100, 2),
                "Recall": round(np.mean(bert_scores_all["recall"]) * 100, 2),
                "fmeasure": round(np.mean(bert_scores_all["f1"]) * 100, 2),
            },
        }
        performance = {**performance, **performance_all}

    print(experiment["experiment_name"])

    performance_df = pd.DataFrame(performance).T
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto", floatfmt=".2f"))
    bleu_score_all = None
    if multiple_answers:
        print("load bleu score all")
        bleu_score_all = bleu_all(
            results[generated_column], results[reference_column], CONFIG
        )
        print("bleu_score_all", bleu_score_all["bleu"])

    performance_bleu = {
        "Bleu ": {
            "bleu_score": round(bleu_score["bleu"] * 100, 2),
            "precisions": round(bleu_score["precisions"][0] * 100, 2),
            "2g": round(bleu_score["precisions"][1] * 100, 2),
            "3g": round(bleu_score["precisions"][2] * 100, 2),
            "4g": round(bleu_score["precisions"][3] * 100, 2),
        }
    }
    if multiple_answers:
        performance_bleu["Bleu All"] = {
            "bleu_score": round(bleu_score_all["bleu"] * 100, 2),
            "precisions": round(bleu_score_all["precisions"][0] * 100, 2),
            "2g": round(bleu_score_all["precisions"][1] * 100, 2),
            "3g": round(bleu_score_all["precisions"][2] * 100, 2),
            "4g": round(bleu_score_all["precisions"][3] * 100, 2),
        }


    performance_df = pd.DataFrame(performance_bleu).T
    print(f"Aggregated metrics for the complete dataset")
    print(tabulate(performance_df, headers="keys", tablefmt="presto", floatfmt=".2f"))

    #performance = performance_bleu | performance
    performance = {**performance_bleu, **performance}
    if args.per_example_scores:
        performance["rougeP_per_example"]=rouge_scores_ov_all["precision"]
        performance["rougeR_per_example"]=rouge_scores_ov_all["recall"]
        performance["rougeF_per_example"]=rouge_scores_ov_all["fmeasure"]

        performance["bleu_per_example"]=bleu_score_all["bleu_per_example"]
        performance["bertP_per_example"]=bert_scores_all["precision"]
        performance["bertR_per_example"]=bert_scores_all["recall"]
        performance["bertf_per_example"]=bert_scores_all["f1"]
    results_file = results_file[:-5] + "_perf_answer.json"

    with open(results_file, "w") as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    main()
