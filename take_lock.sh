#!/bin/bash

# reserve a GPU (take a lock)
srun -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --pty bash
