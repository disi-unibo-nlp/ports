<div align="center"><img src="assets/ports_icon.png" width="10%"> </div>
<h1 align="center"> PORTS </h1>
<h2 align="center">Preference-Optimized Retrievers for Tool Selection with Large Language Models  </h2>

This repository contains the code and datasets for reproducing the experiments described in the paper titled "PORTS: Preference-Optimized Retrievers for Tool Selection with Large Language Models." The paper introduces PORTS, a novel method to fine-tune retrievers that align with the preferences of a frozen Large Language Model (LLM) for tool selection tasks. By optimizing the correlation between retrieval probabilities and downstream performance, PORTS enhances the accuracy of tool selection while maintaining low computational demands. The approach is validated through extensive experiments on six diverse datasets, demonstrating significant improvements in tool selection accuracy compared to existing methods.

<p align="center">
<img src="assets/ports_overview.png" width="70%" height="auto" alt="PORTS Training Overview" class="center">
</p>


## 📎 Table of Contents

- [Model](#model)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Main Accuracy Results](#main-accuracy-results)

## Model

PORTS fine-tunes a retriever model to select the most appropriate tools based on preferences derived from a frozen LLM. The retriever is optimized through a dual loss approach: a perplexity-based preference signal and a contrastive semantic loss. This ensures that the retriever aligns with the LLM's preferences, leading to more accurate tool selection in various scenarios. The model is trained on two encoder architectures, RoBERTa-base and BGE-base, and evaluated using three LLMs with varying levels of expertise in tool usage.

## Dataset

The experiments are conducted on six publicly available datasets, which include APIBench, API-Bank, Octopus-v2, ToolE, BFCL, and ToolBench. These datasets cover a wide range of applications, input modalities, and toolsets, providing a comprehensive evaluation of PORTS' effectiveness. For detailed descriptions and statistics of each dataset, please refer to the Supplementary Material in the paper.

## Quickstart

To get started with reproducing the experiments:

1. **Clone the repository:**
   ```bash
   git clone https://anonymous.4open.science/r/ports-code/
   cd ports-code
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the datasets:**
   The datasets can be downloaded from the following link: [PORTS Datasets](https://anonymous.4open.science/r/ports-data/). Extract the datasets into the `data/` directory.

4. **Run the training script:**
   ```bash
   python train_ports.py --config config.yaml
   ```

5. **Evaluate the model:**
   ```bash
   python evaluate_ports.py --config config.yaml
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
