"""JSONL datasets.

SFT rows: {"prompt": ..., "response": ...}
DPO rows: {"prompt": ..., "chosen": ..., "rejected": ...}

Prompt tokens are masked out of the labels (-100) so loss is only on the response.
"""

import json

import torch
from torch.utils.data import Dataset

IGNORE = -100


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _encode(tokenizer, prompt: str, response: str, max_len: int) -> dict:
    p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (p + r)[:max_len]
    labels = ([IGNORE] * len(p) + r)[:max_len]
    return {"input_ids": ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = [_encode(tokenizer, r["prompt"], r["response"], max_len) for r in _read_jsonl(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.rows = [
            {
                "chosen": _encode(tokenizer, r["prompt"], r["chosen"], max_len),
                "rejected": _encode(tokenizer, r["prompt"], r["rejected"], max_len),
            }
            for r in _read_jsonl(path)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _pad(examples: list[dict], pad_id: int) -> dict:
    n = max(len(e["input_ids"]) for e in examples)
    ids, labels, mask = [], [], []
    for e in examples:
        k = n - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * k)
        labels.append(e["labels"] + [IGNORE] * k)
        mask.append([1] * len(e["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def sft_collate(pad_id: int):
    return lambda batch: _pad(batch, pad_id)


def dpo_collate(pad_id: int):
    # Chosen examples first, then rejected, so one forward pass covers both.
    return lambda batch: _pad([b["chosen"] for b in batch] + [b["rejected"] for b in batch], pad_id)
