#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths and parameters
PYTHON_SCRIPT="/home/molfetta/ports/main/src_dml/train_mnrl.py"
DATASET_NAME="bfcl" # Example dataset, change as needed (e.g., "apibank", "toolbench")


MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" # Example model, change as needed
OUTPUT_DIR="/home/molfetta/ports/main/output/mnrl_retriever_${DATASET_NAME}_$(basename ${MODEL_NAME})_$(date +%Y%m%d_%H%M%S)"



EPOCHS=5
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=16
LR=2e-5
MAX_SEQ_LENGTH=384 # Adjust based on model and data
EVAL_STEPS=500 # Evaluate every 500 steps
SEED=42
# Add other arguments as needed, matching train_mnrl.py

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the training script
echo "Starting MNRL training..."
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME"
echo "Output Dir: $OUTPUT_DIR"

python $PYTHON_SCRIPT \
    --dataset "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --retriever_max_seq_length $MAX_SEQ_LENGTH \
    --lr $LR \
    --scheduler "warmupcosine" \
    --pooling "mean" \
    --warmup_steps 100 \
    --eval_steps $EVAL_STEPS \
    --negatives_per_sample 1 \
    --random_negatives \
    --evaluate_on_test \
    --use_pre_trained_model \
    --push_to_hub \
    --public_model \
    --log_file "$OUTPUT_DIR/training.log" \
    --seed $SEED \
    # --disable_wandb # Uncomment to disable W&B logging
    # --max_train_samples 10000 # Uncomment to limit training samples
    # Add any other specific arguments here

echo "Training finished."
echo "Model and logs saved in: $OUTPUT_DIR"
