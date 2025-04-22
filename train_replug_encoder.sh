#!/bin/bash


OUTPUT_DIR="/home/molfetta/ports/output" 
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --memory="30g" \
    --rm \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    ports_image \
    replug \
    DATASET=toolbench \
    RETRIEVAL_MODEL=BAAI/bge-base-en-v1.5 \
    INFERENCE_MODEL=llama3-8B \
    RETRIEVAL_MAX_SEQ_LEN=512 \
    INFERENCE_MAX_SEQ_LEN=1024 \
    BATCH_SIZE=2 \
    LR=1e-5 \
    SCHEDULER=cosine \
    EPOCHS=1 \
    NUM_RETRIEVED_DOCS=3 \
    GAMMA=0.5 \
    BETA=0.5 \
    EVAL_STEPS=0.2 \
    SEED=42 \
    USE_4BIT=true \
    MAX_TRAIN_SAMPLES=10000 \
    WARMUP_RATIO=0.1 \
    SAVE_PATH=/workspace/output \
    WANDB_PROJECT_NAME=PORTS_AAAI-EMNLP \
    WANDB_RUN_NAME=Replug_toolbench_g3 \
    K_EVAL_VALUES_ACCURACY="1 3 5" \
    K_EVAL_VALUES_NDCG="1 3 5"