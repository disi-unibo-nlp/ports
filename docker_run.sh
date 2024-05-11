#!/bin/bash

SCRIPT_PATH=$1
# docker run --gpus=all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -v ./../mounted:/proj/mounted proj $SCRIPT_PATH

# start one or more containers on the SLURM-allocated GPU
docker run -v ./../mounted:/proj/mounted --rm --gpus device=$CUDA_VISIBLE_DEVICES -it proj $SCRIPT_PATH