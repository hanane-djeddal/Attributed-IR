import os
import sys
import re
from typing import List
from transformers import set_seed

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(ROOT_PATH)

from config import CONFIG

set_seed(CONFIG["zephyr"]["SEED"])


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
        # complete_user_prompt = "\n\n\n".join(fewshot_examples) + "\n\n\n" + user_prompt
        # user_prompt = complete_user_prompt

    print("Example prompt:", user_prompt)
    print("Example System prompt:", system_prompt)
    input_text.append({"role": "user", "content": user_prompt})

    inputs = tokenizer.apply_chat_template(
        input_text, add_generation_prompt=True, return_tensors="pt"
    )

    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=1024,
        temperature=0.7,
        # do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    keyword = "<|assistant|>"
    filetred_answer = answer
    if keyword in answer:
        start_index = answer.index(keyword)
        filetred_answer = answer[start_index + len(keyword) :]
    queries = parse_generated_queries(filetred_answer)
    return queries
