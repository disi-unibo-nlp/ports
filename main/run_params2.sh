#!/bin/bash

LLAMA_TYPE="llama3"
LLAMA_GROQ_TYPE="llama3groq"
CODESTRAL_TYPE="codestral"
LLAMA="/proj/mounted/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
LLAMA_GROQ="/proj/mounted/models/models--Groq--Llama-3-Groq-8B-Tool-Use/snapshots/1d0841d97fef29b98cd9737f5beccf9ea0c8f512"
CODESTRAL="/proj/mounted/models/models--mistralai--Codestral-22B-v0.1/snapshots/8f5fe23af91885222a1563283c87416745a5e212"

BGE="/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
ROBERTA="FacebookAI/roberta-base"

OCTOVER="ToolRetriever/OctopusOverlapping"
OCTNONOVER="ToolRetriever/OctopusNonOverlapping"
TOOLEOVER="ToolRetriever/ToolEOverlapping"
TOOLENONOVER="ToolRetriever/ToolENonOverlapping"
APIBANK="ToolRetriever/APIBank"
APIBENCH="ToolRetriever/APIBench"
BFCL="ToolRetriever/BFCL"
TOOLBENCH="ToolRetriever/ToolBench"


quintuplets=(
    "[PAPER][REDO]_ToolEOverlapping_roberta_llama3 $TOOLEOVER $ROBERTA $LLAMA $LLAMA_TYPE"
    "[PAPER][REDO]_ToolEOverlapping_bge_llama3 $TOOLEOVER $BGE $LLAMA $LLAMA_TYPE"
    "[PAPER][REDO]_APIBank_roberta_llama3 $APIBANK $ROBERTA $LLAMA $LLAMA_TYPE"
    "[PAPER][REDO]_APIBank_roberta_llama3groq $APIBANK $ROBERTA $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    "[PAPER][REDO]_APIBank_roberta_codestral $APIBANK $ROBERTA $CODESTRAL $CODESTRAL_TYPE"
    "[PAPER][REDO]_BFCL_roberta_llama3 $BFCL $ROBERTA $LLAMA $LLAMA_TYPE"
    "[PAPER][REDO]_BFCL_roberta_llama3groq $BFCL $ROBERTA $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    "[PAPER][REDO]_BFCL_roberta_codestral $BFCL $ROBERTA $CODESTRAL $CODESTRAL_TYPE"
)


DOCS_PATH="null"
EVAL_DOCS_PATH="null"
QUERY_COLUMN="query_for_retrieval"
RESPONSE_COLUMN="answer"
BATCH_SIZE=4
NUM_TRAIN_EPOCHS=2
NUM_RETRIEVED_DOCS_PER_QUERY=3
GAMMA_VALUE=0.5
BETA_VALUE=0.5
LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"
DATASET_TYPE="function_calling"

for quintuplet in "${quintuplets[@]}"; do
    # Split the quintuplet into individual variables
    IFS=' ' read -r WANDB_PROJ_NAME DATASET_PATH RETR_MODEL_NAME_OR_PATH INFER_MODEL_NAME_OR_PATH INFER_MODEL_TYPE <<< "$quintuplet"
    python3 /proj/main/main_parsed.py  --dataset_path $DATASET_PATH \
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
                            --modified_loss \
                            --eval_docs_path $EVAL_DOCS_PATH \
                            --log_to_wandb \
                            --wandb_proj_name $WANDB_PROJ_NAME 
done
