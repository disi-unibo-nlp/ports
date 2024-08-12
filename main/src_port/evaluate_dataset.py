import wandb
import math
from tqdm import tqdm
import argparse
import random
import json

from typing import List
from transformers.data.data_collator import DataCollatorMixin
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict

import torch.nn.functional as F

from datasets import (
    Dataset
)

from torch.nn import KLDivLoss

import os
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler,
    set_seed
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk

from tqdm import tqdm


from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed
)
import torch

import logging
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a formatter
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Check if the root logger already has handlers
if not root_logger.handlers:
    # Create a stream handler if no handlers exist
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    # Set the formatter for existing handlers
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

# Create a logger for the current module
logger = logging.getLogger(__name__)


from prompt_templates import (
    pseudo_name_mapping, 
    pseudo_name_instr_mapping, 
    PROMPT_TEMPLATES
)

from utils import (
    compute_similarity,
    compute_embeddings,
    get_gradient_norm,
    encode_query,
    embed_corpus,
    get_ndcg_scores
)

from dataset_helper import (
    DatasetDownloader,
    get_eval_dataloader,
    get_train_dataloader
)

from loss_functions import (
    compute_perplexity,
    get_batch_logps,
    odds_ratio_loss,
    compute_Pr,
    get_perplexity,
    compute_loss
)

from datasets import load_dataset


def evaluate(retr_model,
             eval_dataloader,
             eval_triplets,
             corpus_embeddings,
             device : str =  "cuda", 
             k : int = 3,
             api_corpus : List[str] = None,
             retr_tokenizer : AutoTokenizer = None):
    retr_model.eval()

    ranks = [0 for _ in range(k)]

    ndcg_k_values = [1, 3, 5]
    ndcg_scores = [[] for _ in range(len(ndcg_k_values))]
    num_samples = len(eval_dataloader.dataset)

    with torch.no_grad():
        for bid, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            queries = batch["query"]
            bs = queries["input_ids"].shape[0]


            gold_ids = []
            for _i, i in enumerate(range(bid*bs,bid*bs+bs)):
                pos = eval_triplets[i]["positive"]
                idx = api_corpus.index(pos)
                gold_ids.append(idx)
            

            # Compute query embeddings
            query_embeddings = encode_query(retr_model, queries, device)

            # Compute similarities with the entire corpus
            all_similarities = torch.matmul(query_embeddings, corpus_embeddings.T)  # [bs, num_docs]
            all_similarities = all_similarities.squeeze(0).view(bs,-1)

            # Compute ranks
            _, indices = all_similarities.topk(k, dim=-1, largest=True)

            indices = indices.view(bs,-1)
            gold_ids = torch.tensor(gold_ids).view(1,-1).to(device)

            for _k in range(k):
              #rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.unsqueeze(0).T, dim=-1).sum()
              rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.view(bs,-1), dim=-1).sum()
              ranks[_k] += rank_at_n.item()
            
            # Compute NDCG score
            this_ndcg_scores = get_ndcg_scores(ndcg_k_values, batch, gold_ids.squeeze(0), all_similarities, api_corpus, [eval_triplets[i] for i in range(bid*bs, bid*bs+bs)])
            for i, _ in enumerate(this_ndcg_scores):
                ndcg_scores[i] += this_ndcg_scores[i]

    # Normalize ranks
    ranks = [round(r / num_samples * 100, 3) for r in ranks]
    
    ndcg_score_avg = [0 for _ in range(len(ndcg_k_values))]
    for i, k_val in enumerate(ndcg_k_values):
        ndcg_score_avg[i] = round(sum(ndcg_scores[i]) / num_samples,3)

    return ranks, ndcg_score_avg


def main():
    device="cuda"

    ds = load_dataset("ToolRetriever/ToolBench", "parsed_data")
    GROUP="G3"
    ds = ds["train"].filter(lambda x : x["group"] == GROUP)
    print("Loading data")
    dataset = ds.train_test_split(test_size=0.3, seed=42)
    

    print("Loading model")
    model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)

    print("Loading corpus")
    eval_api_corpus = list(set(dataset["test"]["api_description"]))

    retriever_max_seq_length=512
    eval_batch_size = 512

    print("Get dataloader")
    eval_data_config = {
        "dataset" : dataset, 
        "api_corpus_list" : eval_api_corpus,
        "retrieval_max_length" : retriever_max_seq_length,
        "retrieval_tokenizer" : tokenizer,
        "batch_size" : eval_batch_size
    }
    eval_triplet_dataloader, eval_triplets = get_eval_dataloader(**eval_data_config)

    logger.info("Embedding Tool Corpus")
    eval_corpus_embeddings = embed_corpus(encoder,
                                        tokenizer,
                                        eval_api_corpus,
                                        device,
                                        batch_size=eval_batch_size,
                                        max_length=retriever_max_seq_length)
    eval_corpus_embeddings = eval_corpus_embeddings.to(device)

    print("Starting evaluation")
    ranks, ndcg_scores = evaluate(retr_model=encoder,
                        eval_dataloader=eval_triplet_dataloader,
                        eval_triplets=eval_triplets,
                        corpus_embeddings=eval_corpus_embeddings,
                        device = device, 
                        k = 3,
                        api_corpus = eval_api_corpus,
                        retr_tokenizer = tokenizer)
    
    # Print results
    k_eval=3
    print("\n")
    print("EVALUATION")
    print("*"*50)
    print(">>> R@K")
    for n in range(k_eval):
        print(f"RANK@{n+1}: {ranks[n]:.2f}%")
    print("-"*30)
    print(">>> NDCG@K")
    ndcg_k_values = [1, 3, 5]
    for n, _k in enumerate(ndcg_k_values):
        print(f"NDCG@{_k}: {ndcg_scores[n]:.2f}")
    print("*"*50)
    print("\n")


if __name__ == '__main__':
    main()