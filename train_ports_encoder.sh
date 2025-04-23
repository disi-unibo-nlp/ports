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
    ports \
    DATASET=toolbench \
    INFERENCE_MODEL=llama3-8B \
    RETRIEVAL_MODEL=BAAI/bge-base-en-v1.5 \
    RETRIEVAL_MAX_SEQ_LEN=512 \
    INFERENCE_MAX_SEQ_LEN=1024 \
    PADDING_SIDE=left \
    N_NEGS=3 \
    EVAL_BATCH_SIZE=2 \
    PREPROCESS_BATCH_SIZE=2 \
    LAMBDA_WEIGHT=0.3 \
    WANDB_PROJECT_NAME=PORTS_AAAI-EMNLP \
    WANDB_RUN_NAME=PORTS_toolbench_g3 \
    LOG_FREQ=20 \
    BETA=0.5 \
    GAMMA=0.5 \
    PREF_BETA=1 \
    CORPUS_UPDATES=10 \
    EPOCHS=1 \
    BATCH_SIZE=2 \
    LR=1e-3 \
    SCHEDULER=cosine \
    SEED=42 \
    EVAL_STEPS=0.2 \
    USE_4BIT=true \
    WARMUP_RATIO=0.2 \
    SAVE_STRATEGY=epoch \
    SAVE_DIR=/workspace/output \
    K_EVAL_VALUES_ACCURACY="1 3 5 10 20" \
    K_EVAL_VALUES_NDCG="1 3 5 10 20" \
    MAX_TRAIN_SAMPLES=10000