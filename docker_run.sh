#!/bin/bash

COMMAND=${1:-bash}

# start rag-func-call container
docker run --name proj -v ./main:/proj/main -v ./../mounted:/proj/mounted --gpus device=$CUDA_VISIBLE_DEVICES -it rag-func-call $COMMAND
# docker run --gpus=all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -v ./../mounted:/proj/mounted proj $COMMAND
