import re
from nltk import sent_tokenize

import re
import pandas as pd
import argparse
import os
import sys
import json


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)
MAX_source = 7
MAX_sentences = 20

from config import CONFIG
from tqdm import tqdm
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from src.data.hagrid_dataset_tools import get_attributable_answer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AUTOAIS = "google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def get_source_from_text(passage):
    pattern = re.compile(r"\[\d+(?:,\s*\d+)*\](?:,\s\[\d+(?:,\s*\d+)*\])*")
    number_pattern = re.compile(r"\d+")
    citations = pattern.findall(passage)
    sources = []
    for src in citations:
        int_src = list(set([int(e) for e in number_pattern.findall(src)]))
        sources.extend(int_src)
    return sources


def extract_citation(passage, gold_sentences=False, keep_source_in_sentence=False):
    """
    Extract citations from text. Splits the text into sentences and identifies the citation in each sentence.
    Args:
        gold_sentences : if True uses the split sentences in the Hagrid dataset. Else it tokenizes the answer
        keep_source_in_sentence : Keeps the citations in the sentences
    """
    if gold_sentences:
        sentences = [s["text"] for s in passage["sentences"]]
    else:
        sentences = sent_tokenize(passage)
    sent_src = {}
    pattern = re.compile(r"\[\d+(?:,\s*\d+)*\](?:,\s\[\d+(?:,\s*\d+)*\])*")
    number_pattern = re.compile(r"\d+")
    for i in range(len(sentences)):
        if sentences[i]:
            src_start = pattern.findall(sentences[i])
            flt = ", ".join(src_start)
            src_rest = pattern.findall(sentences[i])
            if src_start and sentences[i][: len(src_start[0])] == src_start[0]:
                if i - 1 in sent_src.keys():
                    int_src = list(
                        set([int(e) for e in number_pattern.findall(src_start[0])])
                    )
                    sent_src[i - 1]["source"] = int_src
                src_rest = pattern.findall(sentences[i][len(src_start[0]) :])

            if not (src_start and flt == sentences[i]):
                int_src = list(
                    set([int(e) for e in number_pattern.findall(" ".join(src_rest))])
                )
                if keep_source_in_sentence:
                    sent_without_src = sentences[i]
                else:
                    sent_without_src = re.sub(pattern, "", sentences[i]).replace(
                        " .", "."
                    )
                sent_src[i] = {
                    "sentence": sent_without_src,
                    "source": int_src,
                }

    return sent_src


def citation_overlap(example, architecture="RTG-vanilla'"):
    """
    Calculates overlap in citations between gold and generated text. The citations must reference the same passages.
    """
    if architecture == "RTG-gold":
        passages_column_name = CONFIG["column_names"]["gold_passages"]
    else:
        passages_column_name = CONFIG["column_names"]["passages"]

    gold_answer = get_source_from_text(example["gold_answer"])
    gold_answer_citations_docids = [
        example[CONFIG["column_names"]["gold_passages"]][i - 1]["docid"]
        for i in gold_answer
        if i <= len(example[CONFIG["column_names"]["gold_passages"]])
    ]
    gold_sources = set(gold_answer_citations_docids)

    gen_answer = get_source_from_text(example["processed_generated_text"])
    gen_answer_citations = [
        example[passages_column_name][i - 1]["docid"]
        for i in gen_answer
        if i <= len(example[[passages_column_name]])
    ]
    gen_sources = set(gen_answer_citations)

    matched = gen_sources.intersection(gold_sources)

    if len(gen_sources):
        precision = len(matched) / len(gen_sources)
    else:
        precision = 0
    if len(gold_sources):
        recall = len(matched) / len(gold_sources)
    else:
        recall = 0

    example["source_recall"] = recall
    example["source_precision"] = precision

    return example


def citation_overlap_gold(example):
    """
    Calculates overlap in citations between gold and generated text. The citations must reference the same passages.
    """
    gold_answer = extract_citation(example["gold_answer"])
    gold_answer_citations = [s["source"] for s in gold_answer.values()]
    gold_sources = set([s for l in gold_answer_citations for s in l])

    gen_answer = extract_citation(example["processed_generated_text"])
    gen_answer_citations = [s["source"] for s in gen_answer.values()]
    gen_sources = set([s for l in gen_answer_citations for s in l])
    matched = gen_sources.intersection(gold_sources)

    if len(gen_sources):
        precision = len(matched) / len(gen_sources)
    else:
        precision = 0
    if len(gold_sources):
        recall = len(matched) / len(gold_sources)
    else:
        recall = 0

    example["source_recall"] = recall
    example["source_precision"] = precision

    return example


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(
        input_text, return_tensors="pt", truncation=True
    ).input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=1024)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_nli_autoais(
    passages,
    answer,
    infer_from_citation=True,
    tokenized_answer=False,
    concat_passages=False,
    sent_passage_infer_scoring="Max",
):
    """
    computes autoais for 'answer'
    """
    sentences = extract_citation(answer, gold_sentences=tokenized_answer)
    sentences = list(sentences.values())[:MAX_sentences]
    score_per_sentence = []
    for sent in sentences:
        sent["source"] = sent["source"][:MAX_source]
        if infer_from_citation:
            if concat_passages:
                all_src_passages = " ".join(
                    [passages[s - 1] for s in sent["source"] if s <= len(passages)]
                )
                if all_src_passages:
                    total_sent_score = _run_nli_autoais(
                        all_src_passages, sent["sentence"]
                    )
                    score_per_sentence.append(total_sent_score)
            else:
                sent_psg_score = []
                for src in sent["source"]:
                    if src <= len(passages):
                        sent_psg_score.append(
                            _run_nli_autoais(passages[src - 1], sent["sentence"])
                        )
                if len(sent_psg_score):
                    if sent_passage_infer_scoring == "Max":
                        total_sent_score = max(sent_psg_score)
                    elif sent_passage_infer_scoring == "Mean":
                        total_sent_score = sum(sent_psg_score) / len(sent_psg_score)
                    score_per_sentence.append(total_sent_score)
        else:  # using passages
            max_score = 0
            for psg in passages:
                psg_score = _run_nli_autoais(psg, sent["sentence"])
                if psg_score > max_score and psg_score >= 1:
                    max_score = psg_score
                    break
            score_per_sentence.append(max_score)

    if score_per_sentence:
        return sum(score_per_sentence) / len(score_per_sentence), sum(
            score_per_sentence
        ) / len(sentences)
    else:
        return 0, 0


def compute_nli_prec_rec_autoais(
    passages,
    answer,
    tokenized_answer=False,
):
    """
    computes autoais for 'answer'
    """

    sentences = extract_citation(answer, gold_sentences=tokenized_answer)
    entail_recall = 0
    sent_mcite_overcite = 0
    entail_prec = 0  # precision
    autoais_log = []
    total_citations = 0
    sentences = list(sentences.values())[:MAX_sentences]
    for sent in sentences:
        sent["source"] = sent["source"][:MAX_source]
        joint_entail = -1  # recall
        all_src_passages = ""
        # citations recall
        if len(sent["source"]) == 0:
            joint_entail = 0
        elif any([s > len(passages) for s in sent["source"]]):
            joint_entail = 0
        else:
            total_citations += len(sent["source"])
            all_src_passages = "\n".join(
                [passages[s - 1] for s in sent["source"] if s <= len(passages)]
            )
            if all_src_passages:
                joint_entail = _run_nli_autoais(all_src_passages, sent["sentence"])
        entail_recall += joint_entail
        autoais_log.append(
            {
                "claim": sent,
                "passage": all_src_passages,
                "model_type": "NLI",
                "model_output": joint_entail,
            }
        )

        # precision
        if joint_entail and len(sent["source"]) > 1:
            for src in sent["source"]:
                if src <= len(passages):
                    solo_nli_res = _run_nli_autoais(passages[src - 1], sent["sentence"])
                if not solo_nli_res:
                    all_src_passages_excluding_current = "\n".join(
                        [passages[s - 1] for s in sent["source"] if s != src]
                    )
                    if all_src_passages_excluding_current:
                        nli_result = _run_nli_autoais(
                            all_src_passages_excluding_current, sent["sentence"]
                        )
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                else:
                    entail_prec += 1
        else:
            entail_prec += joint_entail

    recall = entail_recall / len(sentences)
    precision = entail_prec / total_citations if total_citations > 0 else 0
    return precision, recall


def compute_nli_autoais_dataset(
    dataframe,
    column_names=["quotes", "answer"],
    autoais_citation=True,
    nli_prec_recall=False,
):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            dtype=torch.bfloat16,
        )
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS,
            torch_dtype=torch.bfloat16,
            quantization_config=nf4_config,
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS, use_fast=False)
    measure1 = 0
    measure2 = 0
    for _, row in tqdm(dataframe.iterrows()):
        if nli_prec_recall:
            precision, recall = compute_nli_prec_rec_autoais(
                row[column_names[0]],
                row[column_names[1]],
            )
            measure1 += precision
            measure2 += recall
        else:
            autoais_only_cited_sentences, autoais_all_sentences = compute_nli_autoais(
                row[column_names[0]],
                row[column_names[1]],
                infer_from_citation=autoais_citation,
            )
            measure1 += autoais_only_cited_sentences
            measure2 += autoais_all_sentences
    s1 = measure1 / len(dataframe)
    s2 = measure2 / len(dataframe)
    if nli_prec_recall:
        scores = {"precision": s1, "recall": s2}
    else:
        scores = {"inlcuding all sentences": s2, "only sentences with citation": s1}

    return scores


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
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument(
        "--autoais",
        type=str,
        default="Cit",
        choices=["Cit", "Pssg", "ALCE"],
    )
    args = parser.parse_args()
    experiment = CONFIG["architectures"][args.architecture]
    results_folder = experiment["experiment_path"] + experiment["experiment_name"]
    results_file = (
        args.results_file
        if args.results_file
        else (results_folder + "/" + experiment["results_file"])
    )

    if results_file.endswith(".json"):
        with open(results_file) as f:
            json_dict = json.load(f)
            results = pd.json_normalize(json_dict["data"])
    else:
        if CONFIG["multiple_gold_answers"]:
            results = pd.read_csv(
                results_file,
                converters={
                    CONFIG["column_names"]["passages"]: eval,
                    CONFIG["column_names"]["gold_passages"]: eval,
                    CONFIG["column_names"]["reference"]: eval,
                },
            )
        else:
            results = pd.read_csv(
                results_file,
                converters={
                    CONFIG["column_names"]["passages"]: eval,
                    CONFIG["column_names"]["gold_passages"]: eval,
                },
            )
    print("Evaluating file:", results_file)
    if args.architecture == "RTG-gold":
        passages_column_name = CONFIG["column_names"]["gold_passages"]
    else:
        passages_column_name = CONFIG["column_names"]["passages"]
    #### process generated text
    pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
    results["processed_generated_text"] = results.apply(
        lambda x: x[CONFIG["column_names"]["prediction"]].replace("<|endoftext|>", ""),
        axis=1,
    )
    results["processed_generated_text"] = results[
        "processed_generated_text"
    ].str.replace(pattern, "", regex=True)
    results = results[results[passages_column_name].str.len() > 0]
    if args.overlap:
        results["gold_answer"] = results[CONFIG["column_names"]["reference"]].apply(
            get_attributable_answer
        )
        results = results[results["gold_answer"].str.len() > 0]

        ############# citations overlap:
        results = results.apply(
            lambda x: citation_overlap(x, architecture=args.architecture), axis=1
        )
        print("source_recall", results["source_recall"].mean())
        print("source_precision", results["source_precision"].mean())
    results["quotes"] = results.apply(
        lambda x: [q["text"] for q in x[passages_column_name]], axis=1
    )

    nli_prec_recall = False
    if args.autoais == "ALCE":
        nli_prec_recall = True
        autoais_citation = False
    elif args.autoais == "Cit":
        autoais_citation = True
    else:
        autoais_citation = False

    scores = compute_nli_autoais_dataset(
        results,
        column_names=["quotes", "processed_generated_text"],
        autoais_citation=autoais_citation,
        nli_prec_recall=nli_prec_recall,
    )
    print("Score (sent, src) of generated answer RTG-gen queries:", scores)
    results_file = results_file[:-5] + "_perf_answer_citations.json"
    with open(results_file, "w") as f:
        json.dump(scores, f, indent=4)


main()
