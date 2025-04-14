#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration based on train_ports.sh ---
SEED=42
DATASET_NAME="bfcl" # Options: "bfcl", "apibank", "toolbench", "toole", "octopus", etc.
MODEL_NAME="BAAI/bge-base-en-v1.5" # Options: "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", etc.

EPOCHS=5
TRAIN_BATCH_SIZE=32 # Adjusted for MNRL, potentially higher than PORT
EVAL_BATCH_SIZE=16
PREPROCESSING_BATCH_SIZE=16 # Added for consistency, though less critical for MNRL script structure
LR="2e-5" # Common default for SBERT fine-tuning
MAX_SEQ_LENGTH=512 # Aligned with RETRIEVAL_MAX_SEQ_LEN in train_ports.sh
EVAL_STEPS_FRACTION=0.2 # Evaluate every 20% of steps within an epoch
WARMUP_STEPS=100 # Common default for SBERT
SCHEDULER="warmupcosine"
POOLING="mean" # Common default, can be "cls" or "max"
NEGATIVES_PER_SAMPLE=1 # Default for MNRL, adjust if needed

WANDB_PROJECT_NAME="MNRL_Training" # Specific project for MNRL runs
WANDB_RUN_NAME="MNRL-${DATASET_NAME}-$(basename ${MODEL_NAME})-LR${LR}-E${EPOCHS}"

OUTPUT_DIR="/home/molfetta/ports/main/output/mnrl_retriever_${DATASET_NAME}_$(basename ${MODEL_NAME})_$(date +%Y%m%d_%H%M%S)"
# --- End Configuration ---

# Define the Python script path
PYTHON_SCRIPT="/home/molfetta/ports/main/src_dml/train_mnrl.py"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the training script
echo "Starting MNRL training..."
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Eval Steps Fraction: $EVAL_STEPS_FRACTION"

# Note: W&B logging is handled internally by the script if not disabled
# The script uses wandb.init based on its arguments

python $PYTHON_SCRIPT \
    --dataset "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --preprocessing_batch_size $PREPROCESSING_BATCH_SIZE \
    --retriever_max_seq_length $MAX_SEQ_LENGTH \
    --lr $LR \
    --scheduler "$SCHEDULER" \
    --pooling "$POOLING" \
    --warmup_steps $WARMUP_STEPS \
    --eval_steps $EVAL_STEPS_FRACTION \
    --negatives_per_sample $NEGATIVES_PER_SAMPLE \
    --random_negatives \
    --evaluate_on_test \
    --use_pre_trained_model \
    --push_to_hub \
    --public_model \
    --hub_repo_name "mnrl-${DATASET_NAME}-$(basename ${MODEL_NAME})" \
    --log_file "$OUTPUT_DIR/training.log" \
    --seed $SEED \
    # --disable_wandb # Uncomment to disable W&B logging
    # --max_train_samples 10000 # Uncomment to limit training samples

echo "Training finished."
echo "Model and logs saved in: $OUTPUT_DIR"
