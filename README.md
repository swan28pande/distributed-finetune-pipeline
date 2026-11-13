# Distributed Fine-Tuning Pipeline

A minimal pipeline for fine-tuning LLMs with **SFT** and **DPO** across multiple GPUs and nodes using **PyTorch FSDP**, with **vLLM** evaluation and **Weights & Biases** logging.

```
ftpipe/
  config.py    YAML config + CLI overrides
  data.py      JSONL datasets for SFT and DPO (loss only on response tokens)
  losses.py    sequence log-probs and DPO loss
  train.py     FSDP training loop (SFT or DPO), saves a Hugging Face checkpoint
  evaluate.py  vLLM generation + exact/contains match, logged to W&B
configs/       sft.yaml, dpo.yaml
data/          tiny sample datasets
scripts/       launch.sh (torchrun wrapper)
```

## How it works

- **FSDP:** each transformer block is sharded separately (`FULL_SHARD`, bf16 mixed precision). The final checkpoint is gathered on rank 0 and saved in Hugging Face format.
- **SFT:** standard next-token loss, with the prompt tokens masked out so only the response is trained.
- **DPO:** a frozen, FSDP-sharded copy of the starting model is the reference. Chosen and rejected answers go through the model in one forward pass. Logged to W&B: `loss`, `reward_acc` and `reward_margin`.
- **Eval:** vLLM loads the saved checkpoint, generates answers for each prompt, and scores them against the references.

## Usage

Requires Linux with NVIDIA GPUs.

```bash
pip install -r requirements.txt
wandb login

# SFT on one node
scripts/launch.sh configs/sft.yaml

# DPO, starting from the SFT checkpoint
scripts/launch.sh configs/dpo.yaml

# Override any config value on the command line
scripts/launch.sh configs/sft.yaml model_name=meta-llama/Llama-3.1-8B lr=1e-5

# Multi-node: run on every node
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 scripts/launch.sh configs/sft.yaml

# Evaluate with vLLM
python -m ftpipe.evaluate --model checkpoints/dpo --eval-file data/sample_eval.jsonl
```

## Data format

| Mode | JSONL fields                      |
|------|-----------------------------------|
| SFT  | `prompt`, `response`              |
| DPO  | `prompt`, `chosen`, `rejected`    |
| Eval | `prompt`, `reference` (optional)  |
