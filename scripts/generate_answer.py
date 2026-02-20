import os
import sys
import re
import json
import time
import argparse
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
from src.data.hagrid_dataset_tools import prepare_contexts
from src.generation.llms.llama2 import generate_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="zephyr", choices=["zephyr", "llama"]
    )
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
        "--nb_passages", type=int, default=None, help="Number of support passages"
    )

    parser.add_argument(
        "--retrieved_passages_file",
        type=int,
        default=None,
        help="File containing retrieved passages for RTG-vanilla/ RTG-query-gen setting",
    )

    args = parser.parse_args()

    model_config = CONFIG["langauge_model"][args.model_name]
    experiment = CONFIG["architectures"][args.architecture]

    set_seed(model_config["SEED"])
    exception = False
    results = None
    execution_time = 0
    nb_passages = None
    use_support_doc = experiment["use_context"]

    try:
        model_id = args.model_id if args.model_id else model_config["model_id"]
        print("Loading Model:", model_id)
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
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            cache_dir=model_config["cache_dir"],
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
                if "quotes" in dataframe.columns:
                    dataframe = dataframe.rename(
                        columns={"quotes": CONFIG["column_names"]["gold_passages"]}
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

        if experiment["use_retrieved"]:
            retrieved_passages_file = (
                args.retrieved_passages_file
                if args.retrieved_passages_file
                else experiment["retrieved_passages_file"]
            )
            retrieved_passages = pd.read_csv(
                retrieved_passages_file,
                index_col=[0],
                converters={CONFIG["column_names"]["passages"]: eval},
            )
            retrieved_passages = retrieved_passages.rename(
                columns={"retrieved_passages": CONFIG["column_names"]["passages"]}
            )
            dataset = datasets.Dataset.from_pandas(retrieved_passages)
        nb_passages = experiment["nb_passages"]

        results = []
        if use_support_doc:
            if experiment["citation"]:
                prompt = CONFIG["prompts"]["prompt"]
            else:
                prompt = CONFIG["prompts"]["prompt_without_citation"]
        else:
            prompt = CONFIG["prompts"]["prompot_without_context"]
        start = time.time()
        for idx, row in enumerate(tqdm(dataset)):
            user_prompt = re.sub(
                "\{query\}", row[CONFIG["column_names"]["query"]], prompt["user"]
            )
            if use_support_doc:
                if experiment["use_retrieved"]:
                    passages = row[CONFIG["column_names"]["passages"]][:nb_passages]
                elif args.architecture == "RTG-gold":
                    passages = row[CONFIG["column_names"]["gold_passages"]][
                        :nb_passages
                    ]

                context = prepare_contexts(
                    passages,
                    hagrid_gold=(
                        CONFIG["dataset"] == "HAGRID" and experiment["hagrid_gold"]
                    ),
                    citation=experiment["citation"],
                )
                user_prompt = re.sub("\{context\}", context, user_prompt)
            input_text = [
                {
                    "role": "system",
                    "content": prompt["system"],
                },
                {"role": "user", "content": user_prompt},
            ]

            inputs = tokenizer.apply_chat_template(
                input_text,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if args.model_name == "llama":
                tokens = generate_answer(model, tokenizer, inputs)
            else:
                tokens = model.generate(
                    inputs.to(model.device),
                    max_new_tokens=model_config["max_new_tokens"],
                    temperature=model_config["temperature"],
                    pad_token_id=tokenizer.eos_token_id,
                )

            answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
            if args.model_name == "llama":
                pattern = r"\[INST\][\s\S].*?\[/INST\]"
            else:
                pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
            filtered_answer = answer.replace("<|endoftext|>", "")
            filtered_answer = re.sub(pattern, "", filtered_answer, flags=re.DOTALL)

            if experiment["use_retrieved"]:
                row[CONFIG["column_names"]["passages"]] = row[
                    CONFIG["column_names"]["passages"]
                ][:nb_passages]
            row["output"] = filtered_answer
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
            experiment["experiment_path"] + experiment["experiment_name"]
        )
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
            print("New directory for experiment is created: ", experiment_folder)
        if results is not None:
            exp_config = experiment
            results_df = pd.DataFrame.from_dict(results)
            results_file = (
                args.results_file if args.results_file else experiment["results_file"]
            )
            results_df.to_csv(f"{experiment_folder}/{results_file}", index=False)

            print(
                "Result file:",
                f"{experiment_folder}/{results_file}",
            )

            config_file = f"{experiment_folder}/{experiment['config_file']}"
            exp_config["execution_time"] = str(execution_time) + " minutes"
            exp_config["error"] = exception
            exp_config["args"] = vars(args)
            with open(config_file, "w") as file:
                json.dump(exp_config, file)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
