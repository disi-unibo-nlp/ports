#!/bin/bash

DATASET_PATH="new_dataset_path"
DOCS_PATH="new_docs_path"
RETR_MODEL="new_retr_model"
INFER_MODEL="new_infer_model"
QUERY_COLUMN="new_query"
RESPONSE_COLUMN="new_response"
BATCH_SIZE=16
NUM_EPOCHS=6.0
NUM_DOCS_PER_QUERY=6
GAMMA=2.0
BETA=2.0
LEARNING_RATE=2e-5
LR_SCHEDULER='linear'
MODEL_SAVE_PATH="/new/path/retr_model.pth"

python3 try_args.py \
    --dataset_path $DATASET_PATH \
    --docs_path $DOCS_PATH \
    --retr_model_name_or_path $RETR_MODEL \
    --infer_model_name_or_path $INFER_MODEL \
    --query_column $QUERY_COLUMN \
    --response_column $RESPONSE_COLUMN \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --num_retrieved_docs_per_query $NUM_DOCS_PER_QUERY \
    --gamma_value $GAMMA \
    --beta_value $BETA \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --trained_model_save_path $MODEL_SAVE_PATH