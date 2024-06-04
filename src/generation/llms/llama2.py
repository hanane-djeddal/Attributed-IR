import os
import sys
import transformers
from torch import cuda, bfloat16
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
import re
from typing import List



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


def load_model():
    model_id = CONFIG["llama"]["model_id"]
    print("Loading model: ", model_id)
    running_device_setting = CONFIG["device"]
    if running_device_setting != "CPU":
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
    else:
        bnb_config = None

    # begin initializing HF items, you need an access token
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model_config.do_sample = CONFIG["llama"]["do_sample"]

    print("Loading Tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, cache_dir=CONFIG["llama"]["cache_dir"]
    )
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    # device_map = {
    #     "transformer.word_embeddings": 0,
    #     "transformer.word_embeddings_layernorm": 0,
    #     "lm_head": 0,
    #     "transformer.h": 0,
    #     "transformer.ln_f": 1,
    #     "transformer.embed_tokens.weight": 1,
    #     "model.embed_tokens": 1,
    #     "model.layers": 1,
    #     "model.norm": 1,
    # }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,  # quantization_config,  # bnb_config,  ## comment this param when running on CPU
        device_map="auto",  # device_map,  # "auto",
        cache_dir=CONFIG["llama"]["cache_dir"],
    )
    # model = transformers.AutoModelForCausalLM.from_pretrained(PATH_TO_MODEL)

    # enable evaluation mode to allow model inference
    model.eval()

    print(f"Model loaded on {device}")

    # print("saving models")
    # tokenizer.save_pretrained("./models_cache/llama-2-chat-7b/")
    # model.save_pretrained("./models_cache/llama-2-chat-7b/")
    # model_config.save_pretrained("./models_cache/llama-2-chat-7b/")

    stopping_criteria = StoppingCriteriaList([StopOnTokens(tokenizer)])

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",  # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=CONFIG["llama"][
            "max_new_tokens"
        ],  # max number of tokens to generate in the output
        repetition_penalty=CONFIG["llama"][
            "repetition_penalty"
        ],  # without this output begins repeating
    )
    return generate_text


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
    print(answer)
    if keyword in answer:
        start_index = answer.index(keyword)
        filetred_answer = answer[start_index + len(keyword) :]
    queries = parse_generated_queries(filetred_answer)
    return queries

def parse_generated_queries(answer: str):
    answer = re.sub(r"\d+\.", "", answer)
    quries = answer.split("\n")
    quries = list(filter(None, quries))
    return quries
