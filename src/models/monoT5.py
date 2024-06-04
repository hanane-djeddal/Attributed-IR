import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

# from config import CONFIG


def batch(docs: list, nb: int = 10):
    batches = []
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) == nb:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches


def greedy_decode(model, input_ids, length, attention_mask, return_last_logits=True):
    decode_ids = torch.full(
        (input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long
    ).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat(
            [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
        )
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


class MonoT5:
    def __init__(self, model_path="castorini/monot5-base-msmarco", device=None):
        self.model = self.get_model(model_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
        )
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str, *args, device: str = None, **kwargs
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        return (
            AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            .to(device)
            .eval()
        )

    def rerank(self, query, docs):
        d = self.rescore(query, docs)
        id_ = np.argsort([i["score"] for i in d])[::-1]
        return np.array(d)[id_]

    def rescore(self, query, docs):
        for b in batch(docs, 10):
            with torch.no_grad():
                text = [f'Query: {query} Document: {d["text"]} Relevant:' for d in b]
                model_inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = model_inputs["input_ids"].to(self.device)
                attn_mask = model_inputs["attention_mask"].to(self.device)
                _, batch_scores = greedy_decode(
                    self.model,
                    input_ids=input_ids,
                    length=1,
                    attention_mask=attn_mask,
                    return_last_logits=True,
                )
                batch_scores = batch_scores[
                    :, [self.token_false_id, self.token_true_id]
                ]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(b, batch_log_probs):
                doc["score"] = score  # dont update, only used as Initial with query
        return docs
