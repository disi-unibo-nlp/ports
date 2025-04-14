# Ports Project Makefile
# Provides a clean interface for executing training and evaluation scripts

# Default shell
SHELL := /bin/bash

# Default values
DATASET ?= toolbench
RETRIEVAL_MODEL ?= BAAI/bge-base-en-v1.5
INFERENCE_MODEL ?= llama3-8B
EPOCHS ?= 5
BATCH_SIZE ?= 2
LR ?= 1e-5
SEED ?= 42
EVAL_STEPS ?= 0.2
USE_4BIT ?= true

# Directories
MAIN_DIR := main
SCRIPTS_DIR := $(MAIN_DIR)/scripts
OUTPUT_DIR := $(MAIN_DIR)/output

# Helper functions
define print_header
	@echo "====================================="
	@echo "$(1)"
	@echo "====================================="
	@echo "Dataset: $(DATASET)"
	@echo "Retrieval model: $(RETRIEVAL_MODEL)"
	@echo "Inference model: $(INFERENCE_MODEL)"
	@echo "Epochs: $(EPOCHS)"
	@echo "Batch size: $(BATCH_SIZE)"
	@echo "Learning rate: $(LR)"
	@echo "====================================="
endef

# Default target (help)
.PHONY: help
help:
	@echo "Ports Training Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [options]"
	@echo ""
	@echo "Targets:"
	@echo "  help            - Show this help message"
	@echo "  ports           - Run PORTS training"
	@echo "  replug          - Run RePlug training"
	@echo "  mnrl            - Run MNRL training"
	@echo "  docker          - Run interactive Docker container"
	@echo "  docker-cmd      - Run specific command in Docker container"
	@echo ""
	@echo "Options:"
	@echo "  DATASET=<name>          - Dataset name (default: toolbench)"
	@echo "  RETRIEVAL_MODEL=<name>  - Retrieval model name (default: BAAI/bge-base-en-v1.5)"
	@echo "  INFERENCE_MODEL=<name>  - Inference model name (default: llama3-8B)"
	@echo "  EPOCHS=<number>         - Number of training epochs (default: 5)"
	@echo "  BATCH_SIZE=<number>     - Batch size (default: 2)"
	@echo "  LR=<value>              - Learning rate (default: 1e-5)"
	@echo "  SEED=<number>           - Random seed (default: 42)"
	@echo "  EVAL_STEPS=<number>     - Evaluation steps fraction (default: 0.2)"
	@echo "  USE_4BIT=<true|false>   - Use 4-bit quantization (default: true)"
	@echo ""
	@echo "Examples:"
	@echo "  make ports DATASET=bfcl INFERENCE_MODEL=gemma2-2B"
	@echo "  make replug EPOCHS=10 BATCH_SIZE=4"
	@echo "  make mnrl RETRIEVAL_MODEL=FacebookAI/roberta-base"
	@echo "  make docker"

# PORTS training
.PHONY: ports
ports:
	$(call print_header,"Running PORTS Training")
	@mkdir -p $(OUTPUT_DIR)/ports
	@EPOCHS=$(EPOCHS) \
	DATASET_NAME=$(DATASET) \
	RETRIEVAL_MODEL_NAME=$(RETRIEVAL_MODEL) \
	INFERENCE_MODEL_PSEUDONAME=$(INFERENCE_MODEL) \
	TRAIN_BATCH_SIZE=$(BATCH_SIZE) \
	LR=$(LR) \
	SEED=$(SEED) \
	EVAL_STEPS=$(EVAL_STEPS) \
	LOAD_IN_4BIT=$(USE_4BIT) \
	$(SCRIPTS_DIR)/train_ports.sh

# RePlug training
.PHONY: replug
replug:
	$(call print_header,"Running RePlug Training")
	@mkdir -p $(OUTPUT_DIR)/replug
	@EPOCHS=$(EPOCHS) \
	DATASET_PATH="ToolRetriever/$(DATASET)" \
	RETRIEVAL_MODEL_NAME=$(RETRIEVAL_MODEL) \
	INFERENCE_MODEL_PSEUDONAME=$(INFERENCE_MODEL) \
	TRAIN_BATCH_SIZE=$(BATCH_SIZE) \
	LR=$(LR) \
	SEED=$(SEED) \
	EVAL_STEPS_FRACTION=$(EVAL_STEPS) \
	LOAD_IN_4BIT=$(USE_4BIT) \
	$(SCRIPTS_DIR)/train_replug.sh

# MNRL training
.PHONY: mnrl
mnrl:
	$(call print_header,"Running MNRL Training")
	@mkdir -p $(OUTPUT_DIR)/mnrl
	python $(MAIN_DIR)/train_mnrl.py \
	--dataset $(DATASET) \
	--model_name $(RETRIEVAL_MODEL) \
	--epochs $(EPOCHS) \
	--train_batch_size $(BATCH_SIZE) \
	--lr $(LR) \
	--seed $(SEED) \
	--eval_steps $(EVAL_STEPS) \
	--output_dir $(OUTPUT_DIR)/mnrl/$(DATASET)_$(shell basename $(RETRIEVAL_MODEL))_$(shell date +%Y%m%d_%H%M%S)

# Docker commands
.PHONY: docker
docker:
	@echo "Starting interactive Docker container..."
	$(SCRIPTS_DIR)/docker_run.sh

.PHONY: docker-cmd
docker-cmd:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: Please specify a command with CMD=<your-command>"; \
		exit 1; \
	fi
	@echo "Running command in Docker: $(CMD)"
	$(SCRIPTS_DIR)/docker_run.sh "$(CMD)"

# Clean output directories
.PHONY: clean
clean:
	@echo "Cleaning output directories..."
	rm -rf $(OUTPUT_DIR)/ports/*
	rm -rf $(OUTPUT_DIR)/replug/*
	rm -rf $(OUTPUT_DIR)/mnrl/*
	@echo "Cleaned up output directories."