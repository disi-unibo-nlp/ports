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
- [Quickstart](#quickstart)
- [Main Accuracy Results](#main-accuracy-results)

## Model

**PORTS** fine-tunes a retriever model to select the most appropriate tools based on preferences derived from a frozen LLM. The retriever is optimized through a dual loss approach: a perplexity-based preference signal and a contrastive semantic loss. This ensures the retriever aligns with the LLM's preferences, leading to more accurate tool selection in various scenarios. The model is trained on two encoder architectures, RoBERTa-base and BGE-base, and evaluated using three LLMs with varying levels of expertise in tool usage.

## Dataset

The experiments are conducted on six publicly available datasets, which include APIBench, API-Bank, Octopus-v2, ToolE, BFCL, and ToolBench. These datasets cover various applications, input modalities, and toolsets, comprehensively evaluating **PORTS**' effectiveness. Datasets can be found [here](https://anonymous.4open.science/r/ports-data/).

## Quickstart

To get started with reproducing the experiments:

1. **Clone the repository:**
   ```bash
   git clone https://anonymous.4open.science/r/ports-code/
   cd ports-code
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r build/requirements.txt
   ```

3. **Download the datasets:**
   The datasets can be downloaded from the following link: [PORTS Datasets](https://anonymous.4open.science/r/ports-data/). Extract the datasets into the `data/` directory.

4. **Run the training and evaluation script:**
   ```bash
   #!/bin/bash
   
   SEED=42
   
   RETRIEVAL_MAX_SEQ_LEN=512
   INFERENCE_MAX_SEQ_LEN=1024
   PADDING_SIDE="left"
   
   # Data
   N_NEGATIVES=3
   
   # Eval config
   MAX_K_RECALL=3
   
   # Train Config
   LR="1e-5"
   LR_SCHEDULER="cosine"
   LAMBDA_WEIGHT=0.3
   
   N_EPOCHS=3
   BETA=0.5
   GAMMA=0.5
   CORPUS_UPDATES=500

   
   TRAIN_BATCH_SIZE=2
   EVAL_BATCH_SIZE=2
   PREPROCESS_BATCH_SIZE=8
   
   # Models Configuration
   INFERENCE_MODEL_PSEUDONAME="llama3-8B"
   RETRIEVAL_MODEL_NAME="FacebookAI/roberta-base"
   DATASET_NAME="toole"
   
   python3 main_train_port.py --dataset $DATASET_NAME \
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
                              --n_reembedding_steps $CORPUS_UPDATES \
                              --padding_side $PADDING_SIDE \
                              --lambda_loss $LAMBDA_WEIGHT \
                              --n_neg_examples $N_NEGATIVES \
                              --k_eval $MAX_K_RECALL \
                              --gamma $GAMMA \
                              --beta $BETA \
                              --seed $SEED \
                              --log_freq $LOG_FREQ \
                              --do_train \
                              --do_eval \
                              --eval_strategy "epoch" \
                              --eval_steps 100 \
                              --load_in_4bit                              
   ```

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
