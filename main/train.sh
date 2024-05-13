#!/bin/bash

DATASET_PATH="/proj/mounted/overlapping-functions-dataset"
DOCS_PATH="/proj/mounted/documentation.txt"
RETR_MODEL_NAME_OR_PATH="BAAI/bge-base-zh-v1.5"
INFER_MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
QUERY_COLUMN="query"
RESPONSE_COLUMN="response"
BATCH_SIZE=32
NUM_TRAIN_EPOCHS=3
NUM_RETRIEVED_DOCS_PER_QUERY=3
GAMMA_VALUE=1.0
BETA_VALUE=1.0
LEARNING_RATE=5e-5
LR_SCHEDULER="cosine"
TRAINED_MODEL_SAVE_PATH="/proj/mounted/retr_model.pth"

python3 main.py --dataset_path $DATASET_PATH \
                --docs_path $DOCS_PATH \
                --retr_model_name_or_path $RETR_MODEL_NAME_OR_PATH \
                --infer_model_name_or_path $INFER_MODEL_NAME_OR_PATH \
                --query_column $QUERY_COLUMN \
                --response_column $RESPONSE_COLUMN \
                --batch_size $BATCH_SIZE \
                --num_train_epochs $NUM_TRAIN_EPOCHS \
                --num_retrieved_docs_per_query $NUM_RETRIEVED_DOCS_PER_QUERY \
                --gamma_value $GAMMA_VALUE \
                --beta_value $BETA_VALUE \
                --learning_rate $LEARNING_RATE \
                --lr_scheduler $LR_SCHEDULER \
                --trained_model_save_path $TRAINED_MODEL_SAVE_PATH \
                --quantize \
                --quantization_4bit \
                # --log_to_wandb