#!/bin/bash

DATASET_PATH="/proj/mounted/non-overlapping-functions-dataset"
DOCS_PATH="/proj/mounted/documentation.txt"
RETR_MODEL_NAME_OR_PATH="BAAI/bge-base-en-v1.5"
INFER_MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
QUERY_COLUMN="query"
RESPONSE_COLUMN="response"
# GAMMA_VALUE=1.0
# BETA_VALUE=1.0
LR_SCHEDULER="cosine"
TRAINED_MODEL_SAVE_PATH="/proj/mounted/retr_model.pth"
LEARNING_RATE=1e-5
BATCH_SIZE=8

# LEARNING_RATES=(5e-5 3e-4 1e-3)
# BATCH_SIZES=(2 4 8)
GAMMA_VALUES=(0.1 0.3 0.5)
BETA_VALUES=(0.1 0.3 0.5)

TRAIN_EPOCHS=(3 7 10)
NUM_RETRIEVED_DOCS_PER_QUERY=(2 3 4)

# for LEARNING_RATE in ${LEARNING_RATES[@]}; do
#   for BATCH_SIZE in ${BATCH_SIZES[@]}; do
for GAMMA_VALUE in ${GAMMA_VALUES[@]}; do
    for BETA_VALUE in ${BETA_VALUES[@]}; do
        for NUM_TRAIN_EPOCHS in ${TRAIN_EPOCHS[@]}; do
            for NUM_RETRIEVED_DOCS in ${NUM_RETRIEVED_DOCS_PER_QUERY[@]}; do
                python3 /proj/main/main.py --dataset_path $DATASET_PATH \
                                --docs_path $DOCS_PATH \
                                --retr_model_name_or_path $RETR_MODEL_NAME_OR_PATH \
                                --infer_model_name_or_path $INFER_MODEL_NAME_OR_PATH \
                                --query_column $QUERY_COLUMN \
                                --response_column $RESPONSE_COLUMN \
                                --batch_size $BATCH_SIZE \
                                --num_train_epochs $NUM_TRAIN_EPOCHS \
                                --num_retrieved_docs_per_query $NUM_RETRIEVED_DOCS \
                                --gamma_value $GAMMA_VALUE \
                                --beta_value $BETA_VALUE \
                                --learning_rate $LEARNING_RATE \
                                --lr_scheduler $LR_SCHEDULER \
                                --trained_model_save_path $TRAINED_MODEL_SAVE_PATH \
                                --quantize \
                                --quantization_4bit \
                                --log_to_wandb
            done
        done
    done
done
