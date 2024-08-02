#!/bin/bash

# INFER_MODEL_TYPE="phi3"
# TRAINED_MODEL_SAVE_PATH="/proj/mounted/toole_diff_docs_non_overlap"
# RETR_MODEL_NAME_OR_PATH="/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
# EVAL_DOCS_PATH="/proj/mounted/func-docs/documentation-toole-eval.txt"
# INFER_MODEL_NAME_OR_PATH="mistralai/Codestral-22B-v0.1"
# INFER_MODEL_NAME_OR_PATH="/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
DOCS_PATH="/proj/mounted/func-docs/documentation-octopus.txt"
DATASET_PATH="/proj/mounted/datasets/octopus-overlapping"
RETR_MODEL_NAME_OR_PATH="FacebookAI/roberta-base"
INFER_MODEL_TYPE="phi3"
INFER_MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
QUERY_COLUMN="query"
RESPONSE_COLUMN="response"
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=5
NUM_RETRIEVED_DOCS_PER_QUERY=6
GAMMA_VALUE=0.5
BETA_VALUE=0.5
LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"
if [[ $DATASET_PATH == *"octopus"* ]]; then
    DATASET_TYPE="function_calling"
else
    DATASET_TYPE="tool_selection"
fi


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
                            --quantize \
                            --quantization_4bit \
                            --dataset_type $DATASET_TYPE \
                            --verbose \
                            # --modified_loss \
                            # --log_to_wandb \
                            # --wandb_proj_name "octopus non-overlapping codestral 4-bit quant" \
                            # --eval_docs_path $EVAL_DOCS_PATH \
                            # --trained_model_save_path $TRAINED_MODEL_SAVE_PATH \
