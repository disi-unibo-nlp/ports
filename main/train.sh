#!/bin/bash

# INFER_MODEL_TYPE="phi3"
# TRAINED_MODEL_SAVE_PATH="/proj/mounted/toole_diff_docs_non_overlap"
# INFER_MODEL_NAME_OR_PATH="microsoft/Phi-3-mini-4k-instruct"
# RETR_MODEL_NAME_OR_PATH="/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
# EVAL_DOCS_PATH="/proj/mounted/func-docs/documentation-toole-eval.txt"
DOCS_PATH="/proj/mounted/func-docs/documentation-toole.txt"
DATASET_PATH="/proj/mounted/datasets/toole-single-tool-overlapping"
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
                            --modified_loss \
                            --verbose \
                            # --eval_docs_path $EVAL_DOCS_PATH \
                            # --log_to_wandb \
                            # --wandb_proj_name "toole overlapping phi-3 mini" \
                            # --trained_model_save_path $TRAINED_MODEL_SAVE_PATH \
