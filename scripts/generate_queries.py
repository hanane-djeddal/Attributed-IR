import os
import sys
import re
import json
import time

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

from config import CONFIG

from src.generation.llms.zephyr import generate_queries


def main():
    set_seed(CONFIG["experiment"]["SEED"])
    exception = False
    results = None
    execution_time = 0
    try:
        model_id = CONFIG["experiment"]["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CONFIG["experiment"]["cache_dir"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=CONFIG["experiment"]["cache_dir"],
            trust_remote_code=True,
            device_map="auto",
        )

        ## loading data
        if CONFIG["experiment"]["dataset"] == "HAGRID":
            dataset = datasets.load_dataset(
                "miracl/hagrid",
                split="dev",
                cache_dir=CONFIG["experiment"]["cache_dir"],
            )
        else:
            print("Loading data : ", CONFIG["experiment"]["path"])
            dataset = pd.read_csv(
                CONFIG["experiment"]["data_path"],
                encoding="latin-1",
                converters={"outline": eval, "candidats": eval},
            )

        results = []
        prompt = CONFIG["experiment"]["query_gen_prompt"]
        start = time.time()
        for row in tqdm(dataset):
            answer = None
            if CONFIG["experiment"]["include_answer"]:
                answer = row["answers"][0]["answer"]
            examples = None
            if CONFIG["experiment"]["setting"] == "fewshot":
                examples = CONFIG["experiment"]["fewshot_examples"]
            nb_queries_to_generate = CONFIG["experiment"]["nb_queries_to_generate"]
            nb_shots = CONFIG["experiment"]["nb_shots"]
            queries = generate_queries(
                row["query"],
                model,
                tokenizer,
                prompt,
                include_answer=CONFIG["experiment"]["include_answer"],
                answer=answer,
                fewshot_examples=examples,
                nb_queries_to_generate=nb_queries_to_generate,
                nb_shots=nb_shots,
            )
            results.append(
                {
                    "query": row["query"],
                    "query_id": row["query_id"],
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
            CONFIG["experiment"]["experiment_path"]
            + CONFIG["experiment"]["experiment_name"]
        )
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            print("New directory for experiment is created: ", experiment_folder)
        if results is not None:
            exp_config = CONFIG["experiment"]
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(
                f"{experiment_folder}/{CONFIG['experiment']['results_file']}"
            )
            print(
                "Result file:",
                f"{experiment_folder}/{CONFIG['experiment']['results_file']}",
            )
            config_file = f"{experiment_folder}/{CONFIG['experiment']['config_file']}"
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            with open(config_file, "w") as file:
                json.dump(exp_config, file)
        torch.cuda.empty_cache()


main()
