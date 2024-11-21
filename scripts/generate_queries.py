import os
import sys
import argparse
import json
import time

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed
from torch import cuda, bfloat16
from transformers import BitsAndBytesConfig
import transformers

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.generation.llms.zephyr import generate_queries
from src.generation.llms.llama2 import generate_queries_llama

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="zephyr", choices=["zephyr", "llama"]
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="provide a direct valid model_id from huggingFace, for example 'HuggingFaceH4/zephyr-7b-beta' ",
    )

    parser.add_argument(
        "--load_in_4bits", action="store_true", help="Loading the model in 4bits"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Valid name of dataset if available on Huggingface, for example : 'miracl/hagrid' ",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="If you provide dataset from huggingface, provide the split name: dev, test, train",
    )

    parser.add_argument(
        "--data_path", type=str, default=None, help="Custom path to dataset file"
    )

    parser.add_argument(
        "--nb_queries", type=int, default=None, help="Number of queries to generate"
    )

    args = parser.parse_args()
    model_config = CONFIG["langauge_model"][args.model_name]

    set_seed(model_config["SEED"])
    exception = False
    results = None
    execution_time = 0
    try:
        model_id = args.model_id if args.model_id else model_config["model_id"]
        if args.model_name == "llama" or args.load_in_4bits:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bfloat16,
            )
        else:
            bnb_config = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=model_config["cache_dir"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            cache_dir=model_config["cache_dir"],
            trust_remote_code=True,
            device_map="auto",
        )

        ## loading data
        if args.dataset:
            dataset = datasets.load_dataset(
                args.dataset,
                split=args.split,
                cache_dir=model_config["cache_dir"],
            )
        elif args.data_path or CONFIG["data_path"]:
            if CONFIG["data_path"].endswith(".json"):
                print("Loading data : ", CONFIG["data_path"])
                with open(CONFIG["data_path"]) as f:
                    dataset = json.load(f)
            elif CONFIG["data_path"].endswith(".csv"):
                print("Loading data : ", CONFIG["data_path"])
                dataframe = pd.read_csv(
                    CONFIG["data_path"],
                    encoding="latin-1",
                    converters={
                        CONFIG["column_names"]["passages"]: eval,
                    },
                )
                dataset = dataframe.to_dict("records")
            else:
                print(
                    "Data file needs to be a json or a csv file. If the dataset is a huggingface downloadable dataset please specify it through the 'dataset' argument"
                )
        elif CONFIG["dataset"] == "HAGRID":
            dataset = datasets.load_dataset(
                "miracl/hagrid",
                split=args.split,
                cache_dir=model_config["cache_dir"],
            )
        else:
            print("No dataset provided")

        results = []
        prompt = CONFIG["prompts"]["query_gen_prompt"]
        start = time.time()
        for idx, row in enumerate(tqdm(dataset)):
            answer = None
            if CONFIG["query_generation"]["include_answer"]:
                answer = row[CONFIG["column_names"]["reference"]][0]["answer"]
            examples = None
            if CONFIG["query_generation"]["setting"] == "fewshot":
                examples = CONFIG["query_generation"]["fewshot_examples"]
            nb_queries_to_generate = (
                args.nb_queries
                if args.nb_queries
                else CONFIG["query_generation"]["nb_queries_to_generate"]
            )
            nb_shots = CONFIG["query_generation"]["nb_shots"]
            if args.model_name == "llama":
                queries = generate_queries_llama(
                    row[CONFIG["column_names"]["query"]],
                    model,
                    tokenizer,
                    prompt,
                    include_answer=CONFIG["query_generation"]["include_answer"],
                    answer=answer,
                    fewshot_examples=examples,
                    nb_queries_to_generate=nb_queries_to_generate,
                    nb_shots=nb_shots,
                )
            else:
                queries = generate_queries(
                    row[CONFIG["column_names"]["query"]],
                    model,
                    tokenizer,
                    prompt,
                    include_answer=CONFIG["query_generation"]["include_answer"],
                    answer=answer,
                    fewshot_examples=examples,
                    nb_queries_to_generate=nb_queries_to_generate,
                    nb_shots=nb_shots,
                )
            row["generated_queries"] = queries
            results.append(row)
        end = time.time()

        execution_time = (end - start) / 60
    except:
        exception = True
        print("Exception caught")
        raise
    finally:
        print("Saving experiment")

        experiment_folder = (
            CONFIG["query_generation"]["experiment_path"]
            + CONFIG["query_generation"]["experiment_name"]
        )
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            print("New directory for experiment is created: ", experiment_folder)
        if results is not None:
            exp_config = CONFIG["query_generation"]
            results_df = pd.DataFrame.from_dict(results)
            results_file = (
                args.results_file
                if args.results_file
                else f"{experiment_folder}/{CONFIG['query_generation']['results_file']}"
            )
            results_df.to_csv(
                results_file,
                index=False,
            )
            print("Result file:", results_file)
            config_file = (
                f"{experiment_folder}/{CONFIG['query_generation']['config_file']}"
            )
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            with open(config_file, "w") as file:
                json.dump(exp_config, file)
        torch.cuda.empty_cache()


main()
