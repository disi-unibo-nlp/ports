#!/bin/bash

DEVICE=0
SEED=42

DATASET_NAME="octopus"

INFERENCE_MODEL_PSEUDONAME="llama3-8B"
RETRIEVAL_MODEL_NAME="FacebookAI/roberta-base"

RETRIEVAL_MAX_SEQ_LEN=514
INFERENCE_MAX_SEQ_LEN=256
PADDING_SIDE="left"

# Data
N_NEGS=4

# Train Config
N_EPOCHS=15


LR="1e-4"
LR_SCHEDULER="cosine"

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=4
PREPROCESS_BATCH_SIZE=4

# Loss config
LAMBDA_WEIGHT=0.2
GAMMA=0.5
BETA=0.5

# Eval config
K_ACC_UPPER=3


# Tests
MAX_TRAIN_SAMPLES=200
MAX_EVAL_SAMPLES=200

WANDB_PROJECT_NAME="PORTS_AAAI"
WANDB_RUN_NAME="Oct100_L3_same_ds_over_epochs_padd_left"

LOG_FREQ=20


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
                                                        --load_in_4bit \
                                                        #--max_train_samples $MAX_TRAIN_SAMPLES \
                                                        #--max_eval_samples $MAX_EVAL_SAMPLES