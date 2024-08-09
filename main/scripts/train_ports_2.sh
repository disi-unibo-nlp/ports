#!/bin/bash

DEVICE=1

SEED=42

DATASET_NAME="octopus"

INFERENCE_MODEL_PSEUDONAME="groqLlama3Tool-8B" 
#"llama3-8B"

RETRIEVAL_MODEL_NAME="FacebookAI/roberta-base"
#RETRIEVAL_MODEL_NAME=""BAAI/bge-base-en-v1.5""

RETRIEVAL_MAX_SEQ_LEN=512
#RETRIEVAL_MAX_SEQ_LEN=512

INFERENCE_MAX_SEQ_LEN=512
PADDING_SIDE="left"

# Data
N_NEGS=3

# Train Config
N_EPOCHS=15


LR="1e-4"
LR_SCHEDULER="cosine"

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=4
PREPROCESS_BATCH_SIZE=4

# Loss config
LAMBDA_WEIGHT=0.3

GAMMA=0.5       # Act on Similarities
BETA=0.5        # Act on PPL

# Eval config
K_ACC_UPPER=3


# Tests
MAX_TRAIN_SAMPLES=10
MAX_EVAL_SAMPLES=200

WANDB_PROJECT_NAME="PORTS_AAAI"
WANDB_RUN_NAME="APIBench_L3_roberta_lower_lr"

LOG_FREQ=100


CUDA_VISIBLE_DEVICES=$DEVICE python3 main_train_port.py --dataset $DATASET_NAME \
                                                        --inference_model_name $INFERENCE_MODEL_PSEUDONAME \
                                                        --retrieval_model_name $RETRIEVAL_MODEL_NAME \
                                                        --retriever_max_seq_length $RETRIEVAL_MAX_SEQ_LEN \
                                                        --inference_max_seq_length $INFERENCE_MAX_SEQ_LEN \
                                                        --n_epochs $N_EPOCHS \
                                                        --lr $LR \
                                                        --lr_type $LR_SCHEDULER \
                                                        --train_batch_size $TRAIN_BATCH_SIZE \
                                                        --eval_batch_size $EVAL_BATCH_SIZE \
                                                        --preprocessing_batch_size $PREPROCESS_BATCH_SIZE \
                                                        --padding_side $PADDING_SIDE \
                                                        --lambda_loss $LAMBDA_WEIGHT \
                                                        --n_neg_examples $N_NEGS \
                                                        --k_eval $K_ACC_UPPER \
                                                        --gamma $GAMMA \
                                                        --beta $BETA \
                                                        --seed $SEED \
                                                        --wandb_project_name $WANDB_PROJECT_NAME \
                                                        --wandb_run_name $WANDB_RUN_NAME \
                                                        --log_freq $LOG_FREQ \
                                                        --do_train \
                                                        --do_eval \
                                                        --eval_strategy "epoch" \
                                                        --eval_steps 500\
                                                        --load_in_4bit \
                                                        #--max_train_samples $MAX_TRAIN_SAMPLES \
                                                        #--max_eval_samples $MAX_EVAL_SAMPLES

