"""Evaluate a checkpoint with vLLM.

Eval rows: {"prompt": ..., "reference": ...}  (reference is optional)
Reports exact-match / contains-match against the reference and logs samples to W&B.

    python -m ftpipe.evaluate --model checkpoints/sft --eval-file data/sample_eval.jsonl
"""

import argparse
import json

import wandb
from vllm import LLM, SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--wandb-project", default="distributed-finetune")
    ap.add_argument("--output", default="eval_results.jsonl")
    args = ap.parse_args()

    with open(args.eval_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel, dtype="bfloat16")
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate([r["prompt"] for r in rows], params)

    exact = contains = scored = 0
    results = []
    for row, out in zip(rows, outputs):
        text = out.outputs[0].text.strip()
        ref = row.get("reference")
        if ref is not None:
            scored += 1
            exact += text == ref.strip()
            contains += ref.strip().lower() in text.lower()
        results.append({"prompt": row["prompt"], "reference": ref, "output": text})

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics = {"eval/n": len(rows)}
    if scored:
        metrics |= {"eval/exact_match": exact / scored, "eval/contains_match": contains / scored}
    print(json.dumps(metrics, indent=2))

    wandb.init(project=args.wandb_project, job_type="eval", config=vars(args))
    table = wandb.Table(columns=["prompt", "reference", "output"])
    for r in results:
        table.add_data(r["prompt"], r["reference"], r["output"])
    wandb.log({**metrics, "eval/samples": table})
    wandb.finish()


if __name__ == "__main__":
    main()
