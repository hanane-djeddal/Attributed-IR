import os
import sys
import argparse
import json
import time

import torch
import pandas as pd
from tqdm import tqdm
from torch import cuda, bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed
from transformers import BitsAndBytesConfig
import transformers

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.generation.llms.zephyr import generate_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="zephyr", choices=["zephyr", "llama"]
    )
    args = parser.parse_args()
    model_config = CONFIG["langauge_model"][args.model_name]

    set_seed(model_config["SEED"])
    exception = False
    results = None
    execution_time = 0
    try:
        model_id = model_config["model_id"]
        if args.model_name == "llama":
            bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
            )

            quantization_config=bnb_config
        else:
            quantization_config = None
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
        if CONFIG["dataset"] == "HAGRID":
            dataset = datasets.load_dataset(
                "miracl/hagrid",
                split="dev",
                trust_remote_code=True
            )
        elif CONFIG["data_path"].endswith(".json"):
            print("Loading data : ", CONFIG["data_path"])
            with open(CONFIG["data_path"]) as f:
                dataset = json.load(f)
        else:
            print("Loading data : ", CONFIG["data_path"])
            dataset = pd.read_csv(
                CONFIG["data_path"],
                encoding="latin-1",
                converters={ CONFIG["column_names"]["passages"]: eval,  CONFIG["column_names"]["reference"]: eval},
            )

        results = []
        prompt = CONFIG["prompts"]["query_gen_prompt"]
        start = time.time()
        for index, row in enumerate(tqdm(dataset)):
            answer = None
            if CONFIG["query_generation"]["include_answer"]:
                answer = row[CONFIG["column_names"]["reference"]][0]["answer"]
            examples = None
            if CONFIG["query_generation"]["setting"] == "fewshot":
                examples = CONFIG["query_generation"]["fewshot_examples"]
            nb_queries_to_generate = CONFIG["query_generation"][
                "nb_queries_to_generate"
            ]
            nb_shots = CONFIG["query_generation"]["nb_shots"]
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
                model_name = args.model_name
            )
            results.append(
                {
                    "query": row[CONFIG["column_names"]["query"]],
                    "generated_text": queries,
                }
            )
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
            exp_config = CONFIG['query_generation']
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(
                f"{experiment_folder}/{CONFIG['query_generation']['results_file']}"
            )
            print(
                "Result file:",
                f"{experiment_folder}/{CONFIG['query_generation']['results_file']}",
            )
            config_file = (
                f"{experiment_folder}/{CONFIG['query_generation']['config_file']}"
            )
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            with open(config_file, "w") as file:
                json.dump(exp_config, file)
        torch.cuda.empty_cache()


main()
