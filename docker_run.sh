#!/bin/bash

COMMAND=${1:-bash}

# Load keys
source /home/monaldini/rag-function-calling/keys/keys.sh

# start rag-func-call container
docker run --name proj \
           -v /home/monaldini/rag-function-calling/main:/proj/main \
           -v /home/monaldini/rag-function-calling/mounted:/proj/mounted \
           --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
           -e HF_KEY=$HF_KEY \
           -e WANDB_KEY=$WANDB_KEY \
           -it rag-func-call $COMMAND
        #    rag-func-call $COMMAND