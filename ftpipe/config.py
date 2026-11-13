from dataclasses import dataclass, fields

import yaml


@dataclass
class Config:
    mode: str = "sft"                 # "sft" or "dpo"
    model_name: str = "Qwen/Qwen2.5-0.5B"
    train_file: str = "data/sample_sft.jsonl"
    output_dir: str = "checkpoints/run"
    max_len: int = 1024
    batch_size: int = 4               # per GPU
    grad_accum: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 10
    epochs: int = 1
    max_grad_norm: float = 1.0
    beta: float = 0.1                 # DPO temperature
    seed: int = 42
    log_every: int = 10
    wandb_project: str = "distributed-finetune"
    wandb_run_name: str | None = None


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name: f.type for f in fields(Config)}
    for item in overrides or []:
        key, value = item.split("=", 1)
        raw[key] = yaml.safe_load(value)
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw)
