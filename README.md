<div align="center"><img src="assets/fishing_rod_tool.png" width="10%"> </div>
<h1 align="center"><img src="assets/ports_icon.png" alt="port icon" width="25" height="auto"> PORTS</h1>
<h2 align="center">Preference-Optimized Retrievers for Tool Selection with Large Language Models  </h2>

This repository contains the code and datasets for reproducing the experiments described in the paper titled "**PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models.**" The paper introduces **PORTS**, a novel method to fine-tune retrievers that align with the preferences of a frozen Large Language Model for tool selection tasks. By optimizing the correlation between retrieval probabilities and downstream performance, **PORTS** enhances the accuracy of tool selection while maintaining low computational demands. The approach is validated through extensive experiments on six diverse datasets, demonstrating significant improvements in tool selection accuracy compared to existing methods.

<br/>
<p align="center">
<img src="assets/ports_overview.png" width="80%" height="auto" alt="PORTS Training Overview" class="center">
</p>


## 📎 Table of Contents

- [Model](#model)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Using the Markdown File](#using-the-markdown-file)
- [Main Accuracy Results](#main-accuracy-results)

## Model

**PORTS** fine-tunes a retriever model to select the most appropriate tools based on preferences derived from a frozen LLM. The retriever is optimized through a dual loss approach: a perplexity-based preference signal and a contrastive semantic loss. This ensures the retriever aligns with the LLM's preferences, leading to more accurate tool selection in various scenarios. The model is trained on two encoder architectures, RoBERTa-base and BGE-base, and evaluated using three LLMs with varying levels of expertise in tool usage.

## Dataset

The experiments are conducted on six publicly available datasets, which include APIBench, API-Bank, Octopus-v2, ToolE, BFCL, and ToolBench. These datasets cover various applications, input modalities, and toolsets, comprehensively evaluating **PORTS**' effectiveness. Datasets can be found [here](https://anonymous.4open.science/r/ports-data/).

## Project Structure

The repository is organized with the following structure:

```
/ports/
├── Makefile              # Entry point for all training commands
├── main/                 # Main code directory
│   ├── scripts/          # All training and utility scripts
│   ├── src/              # Source code modules
│   │   ├── port/         # PORTS model implementation
│   │   ├── replug/       # RePlug model implementation  
│   │   ├── dml/          # DML (MNRL) implementation
│   │   └── utils/        # Shared utilities
│   ├── main_train_port.py # Main PORTS training script
│   ├── train_replug.py   # RePlug training script
│   ├── train_mnrl.py     # MNRL training script
│   └── output/           # Output directory for trained models
├── datasets/             # Training and evaluation datasets
└── assets/               # Project assets and documentation
```

## Quickstart

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourorg/ports.git
   cd ports
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r build/requirements.txt
   ```

3. **Download the datasets:**
   The datasets can be downloaded from the following link: [PORTS Datasets](https://anonymous.4open.science/r/ports-data/). Extract the datasets into the `datasets/` directory.

### Using the Makefile

We provide a Makefile that offers a clean interface for running all training operations:

```bash
# View available commands and options
make help

# Train PORTS model
make ports DATASET=toolbench INFERENCE_MODEL=llama3-8B

# Train RePlug model
make replug DATASET=bfcl EPOCHS=10 BATCH_SIZE=4

# Train MNRL model
make mnrl RETRIEVAL_MODEL=FacebookAI/roberta-base

# Start an interactive Docker container
make docker

# Run a specific command in Docker
make docker-cmd CMD="python main/train_mnrl.py --help"

# Clean up output directories
make clean
```

### Training Options

All training commands accept the following options:

- `DATASET`: Dataset name (default: toolbench)
- `RETRIEVAL_MODEL`: Retrieval model name (default: BAAI/bge-base-en-v1.5)
- `INFERENCE_MODEL`: Inference model name (default: llama3-8B)
- `EPOCHS`: Number of training epochs (default: 5)
- `BATCH_SIZE`: Batch size (default: 2)
- `LR`: Learning rate (default: 1e-5)
- `SEED`: Random seed (default: 42)
- `EVAL_STEPS`: Evaluation steps fraction (default: 0.2)
- `USE_4BIT`: Use 4-bit quantization (default: true)

### Direct Script Usage

Alternatively, you can run the training script directly:

```
usage: main_train_port.py [-h]
                       [--dataset {bfcl,apibank,apibench,octopus,octopus-overlap,toole,toole-overlap,toolbench}]
                       [--inference_model_name {llama3-8B,codestral-22B,gemma2-2B,groqLlama3Tool-8B}]
                       [--retrieval_model_name RETRIEVAL_MODEL_NAME]
                       [--retriever_max_seq_length RETRIEVER_MAX_SEQ_LENGTH]
                       [--inference_max_seq_length INFERENCE_MAX_SEQ_LENGTH]
                       [--do_train] [--do_eval] [--load_in_4bit]
                       [--eval_strategy {epoch,steps}]
                       [--eval_steps EVAL_STEPS]
                       [--max_train_samples MAX_TRAIN_SAMPLES]
                       [--max_eval_samples MAX_EVAL_SAMPLES]
                       [--n_reembedding_steps N_REEMBEDDING_STEPS]
                       [--n_epochs N_EPOCHS] [--lr LR] [--lr_type LR_TYPE]
                       [--train_batch_size TRAIN_BATCH_SIZE]
                       [--eval_batch_size EVAL_BATCH_SIZE]
                       [--preprocessing_batch_size PREPROCESSING_BATCH_SIZE]
                       [--padding_side PADDING_SIDE]
                       [--lambda_loss LAMBDA_LOSS]
                       [--n_neg_examples N_NEG_EXAMPLES] [--k_eval K_EVAL]
                       [--gamma GAMMA] [--beta BETA]
                       [--preference_weight PREFERENCE_WEIGHT]
                       [--seed SEED]
                       [--wandb_project_name WANDB_PROJECT_NAME]
                       [--wandb_run_name WANDB_RUN_NAME]
                       [--log_freq LOG_FREQ]
```

For example, to train `RoBERTa-base` on the `ToolE` dataset using **PORTS**:
```bash
python3 main/main_train_port.py --dataset toole \
                          --inference_model_name llama3-8B \
                          --retrieval_model_name FacebookAI/roberta-base \
                          --retriever_max_seq_length 512 \
                          --inference_max_seq_length 1024 \
                          --n_epochs 3 \
                          --lr 1e-5 \
                          --lr_type cosine \
                          --train_batch_size 2 \
                          --eval_batch_size 2 \
                          --preprocessing_batch_size 8 \
                          --n_reembedding_steps 500 \
                          --padding_side left \
                          --lambda_loss 0.3 \
                          --n_neg_examples 3 \
                          --k_eval 3 \
                          --gamma 0.5 \
                          --beta 0.5 \
                          --seed 42 \
                          --do_train \
                          --do_eval \
                          --eval_strategy steps \
                          --eval_steps 500 \
                          --load_in_4bit
```

## Using the Markdown File

In addition to the code and datasets, this repository includes a comprehensive Markdown documentation that can help you understand and use the PORTS framework effectively. Here's how to use it:

### Reading the Markdown Documentation

1. **Viewing in GitHub**: 
   The README.md file is automatically rendered by GitHub's web interface. Simply navigate to the repository's main page to view it with proper formatting and styling.

2. **Local Viewing Options**:
   - Use any text editor to open the README.md file
   - For better rendering, use a Markdown viewer like:
     - VS Code with the Markdown Preview extension
     - Typora, a dedicated Markdown editor
     - GitHub Desktop's built-in Markdown preview

### Documentation Structure

The Markdown documentation is organized into clear sections:

- **Model Description**: Technical details about PORTS architecture
- **Dataset Information**: Details about supported datasets and their structure
- **Project Structure**: Overview of code organization
- **Quickstart Guide**: Step-by-step instructions for getting started
- **Training Options**: Comprehensive list of parameters and configurations
- **Results**: Performance metrics and comparisons

### Modifying the Documentation

If you need to update the documentation:

1. Edit the README.md file using any text editor
2. Follow standard Markdown syntax:
   - `#` for headers (more `#` means smaller headers)
   - `*` or `-` for bullet points
   - `|` for table columns
   - `` ``` `` for code blocks (specify language after opening backticks)
   - `[text](url)` for links
   - `![alt text](image_path)` for images

3. Use HTML tags for advanced formatting:
   - `<div align="center">` for centering content
   - `<img src="..." width="...">` for image sizing
   - `<br/>` for line breaks

4. Preview changes using a Markdown viewer before committing

### Extending the Documentation

When adding new features or models to the project, be sure to update:

1. The Table of Contents section
2. The relevant documentation section
3. Any code examples or parameter lists
4. Training and evaluation result tables

Remember to maintain consistent formatting and style to ensure readability.

## Main Accuracy Results

PORTS demonstrates significant improvements in tool selection accuracy across all evaluated datasets. The table below summarizes the main accuracy results, including improvements over the baseline models:

| Dataset     | Encoder       | LLM                    | Recall@1 (%) | Recall@3 (%) | NDCG@5 (%) | Recall Improvement (%) | NDCG Improvement (%) |
|-------------|---------------|------------------------|--------------|--------------|------------|------------------------|-----------------------|
| ToolBench   | RoBERTa-base  | LLAMA3-GROQ-8B-Tool-Use | 49.84        | 70.80        | 65.32      | +4.52                  | +4.37                 |
| API-Bank    | BGE-base      | LLAMA3-GROQ-8B-Tool-Use | 59.12        | 81.50        | 76.10      | +2.83                  | +4.80                 |
| APIBench    | RoBERTa-base  | LLAMA3-8B              | 21.50        | 30.53        | 26.78      | +12.76                 | +13.55                |
| BFCL        | BGE-base      | LLAMA3-GROQ-8B-Tool-Use | 67.20        | 78.10        | 73.10      | +1.03                  | +0.79                 |
| ToolE       | RoBERTa-base  | LLAMA3-8B              | 74.60        | 86.80        | 83.55      | +16.09                 | +13.77                |
| Octopus-v2  | BGE-base      | LLAMA3-8B              | 97.50        | 100          | 100        | +2.50                  | +2.00                 |

For more detailed results, including ablation studies and out-of-domain performance, please refer to the paper.
