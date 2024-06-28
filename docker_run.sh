#!/bin/bash

COMMAND=${1:-bash}

# Load keys
source /home/monaldini/rag-function-calling/keys/keys.sh

CUDA_VISIBLE_DEVICES=1

# start rag-func-call container
docker run --name proj3 \
           -v /home/monaldini/rag-function-calling/main:/proj/main \
           -v /home/monaldini/rag-function-calling/mounted:/proj/mounted \
           --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
           -e HF_KEY=$HF_KEY \
           -e WANDB_KEY=$WANDB_KEY \
           -it rag-func-call $COMMAND
        #    -p 7860:7860 \
        #    rag-func-call $COMMAND