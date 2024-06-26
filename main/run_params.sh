#!/bin/bash

# INFER_MODEL_NAME_OR_PATH="microsoft/Phi-3-small-8k-instruct"
# INFER_MODEL_TYPE="phi3"
# DATASET_PATH="/proj/mounted/datasets/toole-single-tool-dataset"
# RETR_MODEL_NAME_OR_PATH="/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
DATASET_PATH="/proj/mounted/datasets/non-overlapping-functions-dataset-no-ir"
DOCS_PATH="/proj/mounted/func-docs/documentation.txt"
RETR_MODEL_NAME_OR_PATH="FacebookAI/roberta-base"
INFER_MODEL_NAME_OR_PATH="/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
INFER_MODEL_TYPE="llama3"
QUERY_COLUMN="query"
RESPONSE_COLUMN="response"
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=10
NUM_RETRIEVED_DOCS_PER_QUERY=3
GAMMA_VALUE=1
BETA_VALUE=1
LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"
TRAINED_MODEL_SAVE_PATH="/proj/mounted/retr_model.pth"
BETA_VALUES=(0.3 0.5 1)
GAMMA_VALULES=(0.3 0.5 1)
NUM_RETRIEVED_DOCS_PER_QUERY_VALUES=(3 6 9)
NUM_TRAIN_EPOCHS_VALUES=(5 10 15)

for BETA_VALUE in "${BETA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALULES[@]}"; do
        for NUM_RETRIEVED_DOCS_PER_QUERY in "${NUM_RETRIEVED_DOCS_PER_QUERY_VALUES[@]}"; do
            for NUM_TRAIN_EPOCHS in "${NUM_TRAIN_EPOCHS_VALUES[@]}"; do
                python3 /proj/main/main.py  --dataset_path $DATASET_PATH \
                            --docs_path $DOCS_PATH \
                            --retr_model_name_or_path $RETR_MODEL_NAME_OR_PATH \
                            --infer_model_name_or_path $INFER_MODEL_NAME_OR_PATH \
                            --infer_model_type $INFER_MODEL_TYPE \
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
                            --wandb_proj_name "octopus non-overlapping modified loss2"
                            # --quantization_4bit \
            done
        done
    done
done