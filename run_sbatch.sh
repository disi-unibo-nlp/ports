#!/bin/bash


# sbatch  -N 1 \
#         --gpus=nvidia_geforce_rtx_3090:1 \
#         -w faretra \
#         train_mnrl_encoder.sh 

sbatch  -N 1 \
        --gpus=nvidia_geforce_rtx_3090:1 \
        -w deeplearn2 \
        train_ports_encoder.sh 


# sbatch  -N 1 \
#         --gpus=nvidia_geforce_rtx_3090:1 \
#         -w faretra \
#         train_replug_encoder.sh