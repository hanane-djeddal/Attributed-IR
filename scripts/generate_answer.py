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
from src.data.hagrid_dataset_tools import prepare_contexts


def format_support_passages(passages_text, passages_ids):
    passages = []
    for i in range(len(passages_text)):
        passages.append(
            {"idx": i + 1, "docid": passages_ids[i], "text": passages_text[i]}
        )
    return passages


def main():
    set_seed(CONFIG["experiment"]["SEED"])
    exception = False
    results = None
    execution_time = 0
    nb_passages = None
    use_support_doc = CONFIG["experiment"]["use_context"]

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
                "miracl/hagrid", split="dev", cache_dir=CONFIG["zephyr"]["cache_dir"]
            )
        else:
            print("Loading data : ", CONFIG["experiment"]["data_path"])
            dataset = pd.read_csv(
                CONFIG["data"]["path"],
                encoding="latin-1",
                converters={"outline": eval, "candidats": eval},
            )

        if CONFIG["experiment"]["use_retrieved"]:
            df_hagrid = dataset.to_pandas()
            retrieved_passages = pd.read_csv(
                CONFIG["experiment"]["retrieved_passages_file"],
                index_col=[0],
                converters={"retrieved_ids": eval, "retrieved_passages": eval},
            )
            retrieved_passages = retrieved_passages[
                ["query", "retrieved_passages", "retrieved_ids"]
            ]
            merged_df_hagrid = df_hagrid.merge(retrieved_passages, on="query")
            merged_df_hagrid = merged_df_hagrid.rename(
                columns={"quotes": "gold_quotes", "retrieved_passages": "quotes"}
            )
            dataset = datasets.Dataset.from_pandas(merged_df_hagrid)
            nb_passages = CONFIG["experiment"]["nb_passages"]

        results = []
        if use_support_doc:
            if CONFIG["experiment"]["citation"]:
                prompt = CONFIG["prompts"]["prompt"]
            else:
                prompt = CONFIG["prompts"]["prompt_without_citation"]
        else:
            prompt = CONFIG["prompts"]["prompot_without_context"]
        start = time.time()

        for row in tqdm(dataset):
            context = prepare_contexts(
                row["quotes"][:nb_passages],
                retrieved=CONFIG["experiment"]["use_retrieved"],
                citation=CONFIG["experiment"]["citation"],
            )
            user_prompt = re.sub("\{query\}", row["query"], prompt["user"])
            if use_support_doc:
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

            tokens = model.generate(
                inputs.to(model.device),
                max_new_tokens=4069,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

            answer = tokenizer.decode(tokens[0], skip_special_tokens=True)

            pattern = r"<\|system\|>[\s\S]*?<\|assistant\|>\n"
            filtered_answer = answer.replace("<|endoftext|>", "")
            filtered_answer = re.sub(pattern, "", filtered_answer)

            if CONFIG["experiment"]["use_retrieved"]:
                passages = format_support_passages(row["quotes"], row["retrieved_ids"])
                results.append(
                    {
                        "query": row["query"],
                        "generated_text": filtered_answer,
                        "gold_truth": row["answers"],
                        "quotes": passages[:nb_passages],
                        "gold_quotes": row["gold_quotes"],
                    }
                )
            else:
                results.append(
                    {
                        "query": row["query"],
                        "generated_text": filtered_answer,
                        "gold_truth": row["answers"],
                        "gold_quotes": row["quotes"][:nb_passages],
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


if __name__ == "__main__":
    main()
