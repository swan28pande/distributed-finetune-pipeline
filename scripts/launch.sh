#!/usr/bin/env bash
# Launch training on one or more nodes.
#   Single node:  scripts/launch.sh configs/sft.yaml
#   Multi-node:   run on every node with NNODES, NODE_RANK and MASTER_ADDR set.
set -euo pipefail

CONFIG=${1:?usage: launch.sh <config.yaml> [key=value ...]}
shift

torchrun \
  --nnodes="${NNODES:-1}" \
  --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  -m ftpipe.train "$CONFIG" "$@"
