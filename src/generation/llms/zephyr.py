import os
import sys
import re
from typing import List
from transformers import set_seed
from torch import cuda, bfloat16
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(ROOT_PATH)

from config import CONFIG

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_list = ["\nHuman:", "\n```\n"]

        stop_token_ids = [self.tokenizer(x)["input_ids"] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        for stop_ids in stop_token_ids:
            if input_ids.shape[1] >= stop_ids.shape[0]:
                if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                    return True
        return False

def parse_generated_queries(answer: str):
    answer = re.sub(r"\d+\.", "", answer)
    quries = answer.split("\n")
    quries = list(filter(None, quries))
    return quries


def generate_queries(
    query: str,
    model,
    tokenizer,
    prompt,
    include_answer=False,
    answer: str = None,
    fewshot_examples: List = None,
    nb_queries_to_generate: int = 4,
    nb_shots: int = None,
    model_name ="zephyr",
) -> List[str]:
    if include_answer and answer:
        user_prompt = re.sub("\{query\}", query, prompt["user_with_answer"])
        user_prompt = re.sub("\{answer\}", answer, user_prompt)
    else:
        user_prompt = re.sub("\{query\}", query, prompt["user"])

    system_prompt = re.sub(
        "\{nb_queries\}", str(nb_queries_to_generate), prompt["system"]
    )
    input_text = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]
    if fewshot_examples:
        formatted_examples = []
        selected_examples = fewshot_examples[:nb_shots]
        # selected_examples.reverse()
        for example in selected_examples:
            formatted_example = (
                "QUESTION: " + example["user_query"] + " \n\nSUGGESTED QUERIES: \n"
            )

            if len(example["generated_queries"]) >= nb_queries_to_generate:
                selected_queries = example["generated_queries"][:nb_queries_to_generate]
            else:
                selected_queries = example["generated_queries"]
            # selected_queries.reverse()
            for i in range(len(selected_queries)):
                formatted_example = (
                    formatted_example + str(i + 1) + ". " + selected_queries[i] + " \n"
                )

            formatted_examples.append(formatted_example)
        complete_user_prompt = "\n\n".join(formatted_examples) + "\n\n" + user_prompt
        user_prompt = complete_user_prompt

    # print("Example prompt:", user_prompt)
    # print("Example System prompt:", system_prompt)
    input_text.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        input_text, add_generation_prompt=False, return_tensors="pt"
    )
    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=CONFIG["langauge_model"]["llama"]["max_new_tokens"],
        temperature=CONFIG["langauge_model"]["llama"]["temperature"],
        # do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if model_name == "zephyr":
        keyword = "<|assistant|>"
    filetred_answer = answer
    if keyword in answer:
        start_index = answer.index(keyword)
        filetred_answer = answer[start_index + len(keyword) :]
    queries = parse_generated_queries(filetred_answer)
    return queries
