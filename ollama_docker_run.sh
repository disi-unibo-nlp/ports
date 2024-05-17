#!/bin/bash

COMMAND=${1:-bash}

# Load keys
source ~/.keys/keys.sh

# start rag-func-call container
docker run --name ollama \
           -v ./main:/proj/main \
           -v ./../mounted:/proj/mounted \
           --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
           -e HF_KEY=$HF_KEY \
           -e WANDB_KEY=$WANDB_KEY \
           -it ollama/ollama