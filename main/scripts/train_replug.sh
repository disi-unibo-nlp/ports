#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration based on train_ports.sh ---
SEED=42

# Models (Map pseudo-names to actual paths/IDs used by train_replug.py)
# Note: train_replug.py uses args like retr_model_name_or_path, infer_model_name_or_path, infer_model_type
RETRIEVAL_MODEL_NAME="BAAI/bge-base-en-v1.5"
INFERENCE_MODEL_PSEUDONAME="llama3-8B" # Options: "llama3-8B", "codestral-22B", "gemma2-2B", "groqLlama3Tool-8B"

# Map pseudo-name to actual model path/ID and type
case $INFERENCE_MODEL_PSEUDONAME in
  "llama3-8B")
    INFERENCE_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    INFERENCE_MODEL_TYPE="llama3"
    ;;
  "codestral-22B")
    INFERENCE_MODEL_NAME="mistralai/Codestral-22B-v0.1"
    INFERENCE_MODEL_TYPE="codestral" # Assuming 'codestral' is a valid type in PROMPT_TEMPLATES
    ;;
  "gemma2-2B")
    INFERENCE_MODEL_NAME="google/gemma-2-2b-it"
    INFERENCE_MODEL_TYPE="gemma" # Assuming 'gemma' is a valid type
    ;;
  "groqLlama3Tool-8B")
    INFERENCE_MODEL_NAME="Groq/Llama-3-Groq-8B-Tool-Use"
    INFERENCE_MODEL_TYPE="llama3groq" # Assuming 'llama3groq' is a valid type
    ;;
  *)
    echo "Error: Unknown inference model pseudo-name: $INFERENCE_MODEL_PSEUDONAME"
    exit 1
    ;;
esac

RETRIEVAL_MAX_SEQ_LEN=512 # Should match retriever capability
# INFERENCE_MAX_SEQ_LEN is not directly used as arg in train_replug.py, but implied by model

# Data & Training Params
DATASET_PATH="ToolRetriever/ToolBench" # Example: "ToolRetriever/ToolBench", "ToolRetriever/BFCL", "ToolRetriever/APIBench"
QUERY_COLUMN="query_for_retrieval" # Adjust based on dataset
RESPONSE_COLUMN="answer" # Adjust based on dataset

LR="1e-5"
LR_SCHEDULER="cosine" # Options: "cosine", "linear", etc. (matching get_scheduler)

TRAIN_BATCH_SIZE=2 # Keep low due to inference model memory usage
# EVAL_BATCH_SIZE is handled internally by evaluator in train_replug.py

EPOCHS=5
NUM_RETRIEVED_DOCS=5 # 'k' in train_replug.py

# Loss config (REPLUG specific)
GAMMA=0.5 # Temperature for Pr softmax
BETA=0.5  # Temperature for Q softmax

# Eval config
EVAL_STEPS_FRACTION=0.2 # Evaluate every 20% of steps

# W&B
WANDB_PROJECT_NAME="REPLUG_Training" # Specific project for REPLUG runs
WANDB_RUN_NAME="REPLUG-$(basename ${DATASET_PATH})-R_$(basename ${RETRIEVAL_MODEL_NAME})-I_${INFERENCE_MODEL_PSEUDONAME}-LR${LR}"

# Output
SAVE_PATH="/home/molfetta/ports/main/output/replug_retriever_${DATASET_NAME}_$(basename ${RETRIEVAL_MODEL_NAME})_$(date +%Y%m%d_%H%M%S)"

# Quantization
LOAD_IN_4BIT="true" # Corresponds to --load_in_4bit in train_ports.sh -> --quantization_4bit in train_replug.py

# --- End Configuration ---

# Define the Python script path
PYTHON_SCRIPT="/home/molfetta/ports/main/replug/train_replug.py"

# Create output directory if save path is specified and used for saving
mkdir -p $SAVE_PATH

# Run the training script
echo "Starting REPLUG training..."
echo "Dataset Path: $DATASET_PATH"
echo "Retrieval Model: $RETRIEVAL_MODEL_NAME"
echo "Inference Model: $INFERENCE_MODEL_NAME (Type: $INFERENCE_MODEL_TYPE)"
echo "Output Dir: $SAVE_PATH"
echo "Eval Steps Fraction: $EVAL_STEPS_FRACTION"

# Construct args for HfArgumentParser
# Note: Boolean flags are passed without values if true, omitted if false
QUANTIZE_ARGS=""
if [ "$LOAD_IN_4BIT" = "true" ]; then
  QUANTIZE_ARGS="--quantize --quantization_4bit"
fi

python $PYTHON_SCRIPT \
    --retr_model_name_or_path "$RETRIEVAL_MODEL_NAME" \
    --infer_model_name_or_path "$INFERENCE_MODEL_NAME" \
    --infer_model_type "$INFERENCE_MODEL_TYPE" \
    --dataset_path "$DATASET_PATH" \
    --query_column "$QUERY_COLUMN" \
    --response_column "$RESPONSE_COLUMN" \
    --num_train_epochs $EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LR \
    --lr_scheduler "$LR_SCHEDULER" \
    --num_retrieved_docs_per_query $NUM_RETRIEVED_DOCS \
    --gamma_value $GAMMA \
    --beta_value $BETA \
    --seed $SEED \
    --log_to_wandb \
    --wandb_proj_name "$WANDB_RUN_NAME" \
    --trained_model_save_path "$SAVE_PATH" \
    --retr_max_seq_length $RETRIEVAL_MAX_SEQ_LEN \
    --eval_steps $EVAL_STEPS_FRACTION \
    $QUANTIZE_ARGS \
    # --verbose # Uncomment for more detailed logging

echo "Training finished."
echo "Model saved in: $SAVE_PATH"

