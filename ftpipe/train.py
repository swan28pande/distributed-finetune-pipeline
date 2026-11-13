"""FSDP training entrypoint for SFT and DPO.

Launch with torchrun, e.g.:
    torchrun --nproc_per_node=8 -m ftpipe.train configs/sft.yaml
"""

import argparse
import functools
import math
import os

import torch
import torch.distributed as dist
import wandb
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import Config, load_config
from .data import DPODataset, SFTDataset, dpo_collate, sft_collate
from .losses import dpo_loss, sequence_logprobs

BF16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)


def is_main() -> bool:
    return dist.get_rank() == 0


def wrap_fsdp(model: torch.nn.Module, device: int) -> FSDP:
    """Shard each transformer block (as named in the HF model's _no_split_modules)."""
    names = set(getattr(model, "_no_split_modules", None) or [])
    layer_cls = {type(m) for m in model.modules() if type(m).__name__ in names}
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_cls)
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=BF16,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def load_model(cfg: Config, device: int) -> FSDP:
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    return wrap_fsdp(model, device)


def save(model: FSDP, tokenizer, cfg: Config) -> None:
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
        state = model.state_dict()
    if is_main():
        os.makedirs(cfg.output_dir, exist_ok=True)
        model.module.save_pretrained(cfg.output_dir, state_dict=state)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"saved checkpoint to {cfg.output_dir}")
    dist.barrier()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.mode == "sft":
        dataset, collate = SFTDataset(cfg.train_file, tokenizer, cfg.max_len), sft_collate(tokenizer.pad_token_id)
    elif cfg.mode == "dpo":
        dataset, collate = DPODataset(cfg.train_file, tokenizer, cfg.max_len), dpo_collate(tokenizer.pad_token_id)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate, drop_last=True)

    model = load_model(cfg, local_rank)
    ref_model = None
    if cfg.mode == "dpo":
        ref_model = load_model(cfg, local_rank).eval()
        ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = math.ceil(len(loader) / cfg.grad_accum) * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    if is_main():
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            batch = {k: v.cuda(local_rank) for k, v in batch.items()}
            stats = {}
            if cfg.mode == "sft":
                loss = model(**batch).loss
            else:
                inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
                policy_logps = sequence_logprobs(model(**inputs).logits.float(), batch["labels"])
                with torch.no_grad():
                    ref_logps = sequence_logprobs(ref_model(**inputs).logits.float(), batch["labels"])
                loss, stats = dpo_loss(policy_logps, ref_logps, cfg.beta)

            (loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum != 0:
                continue

            grad_norm = model.clip_grad_norm_(cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0:
                reduced = loss.detach().clone()
                dist.all_reduce(reduced, op=dist.ReduceOp.AVG)
                if is_main():
                    metrics = {"train/loss": reduced.item(), "train/lr": scheduler.get_last_lr()[0],
                               "train/grad_norm": grad_norm.item(), "epoch": epoch,
                               **{f"train/{k}": v for k, v in stats.items()}}
                    wandb.log(metrics, step=step)
                    print(f"step {step}/{total_steps} loss {metrics['train/loss']:.4f}")

    save(model, tokenizer, cfg)
    if is_main():
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
