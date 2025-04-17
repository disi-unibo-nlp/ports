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
    mnrl \
    DATASET=toolbench \
    RETRIEVAL_MODEL=BAAI/bge-base-en-v1.5 \
    BATCH_SIZE=32 \
    EVAL_BATCH_SIZE=16 \
    PREPROCESS_BATCH_SIZE=16 \
    LR=2e-5 \
    RETRIEVAL_MAX_SEQ_LEN=512 \
    EVAL_STEPS=0.2 \
    WARMUP_RATIO=0.1 \
    SCHEDULER=warmupcosine \
    POOLING=mean \
    NEGATIVES_PER_SAMPLE=1 \
    SEED=42 \
    MAX_TRAIN_SAMPLES=1000 \
    OUTPUT_DIR=/workspace/output \
    K_EVAL_VALUES_ACCURACY="1 3 5 10" \
    K_EVAL_VALUES_NDCG="1 3 5 10" \
    EPOCHS=5