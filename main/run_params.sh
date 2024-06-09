#!/bin/bash

DATASET_PATH="/proj/mounted/overlapping-functions-dataset-no-ir-no-open"
DOCS_PATH="/proj/mounted/documentation-no-open.txt"
RETR_MODEL_NAME_OR_PATH="/proj/mounted/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
INFER_MODEL_NAME_OR_PATH="/proj/mounted/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
QUERY_COLUMN="query"
RESPONSE_COLUMN="response"
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=50
NUM_RETRIEVED_DOCS_PER_QUERY=3
LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"
TRAINED_MODEL_SAVE_PATH="/proj/mounted/retr_model.pth"

GAMMA_VALUES=(0.3 0.5 1)

for GAMMA_VALUE in ${GAMMA_VALUES[@]}; do
    BETA_VALUE=${GAMMA_VALUE}
        python3 /proj/main/main-forward.py --dataset_path $DATASET_PATH \
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
                --log_to_wandb \
                --wandb_proj_name "no_open_door_func test on train"
done

