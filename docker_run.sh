#!/bin/bash

COMMAND=${1:-bash}

# Load keys
source /home/monaldini/rag-function-calling/keys/keys.sh

# Determine the container name
CONTAINER_NAME="proj"
n=1

while [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; do
    n=$((n+1))
    CONTAINER_NAME="proj${n}"
done

# start rag-func-call container
docker run --name $CONTAINER_NAME \
           -v /home/monaldini/rag-function-calling/main:/proj/main \
           -v /home/monaldini/rag-function-calling/mounted:/proj/mounted \
           --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
           -e HF_KEY=$HF_KEY \
           -e WANDB_KEY=$WANDB_KEY \
           rag-func-call $COMMAND
        #    -it rag-func-call $COMMAND
        #    -p 7860:7860 \