import wandb
import math
from tqdm import tqdm
import argparse
import random
import json

from typing import List
from transformers.data.data_collator import DataCollatorMixin
from torch.utils.data import DataLoader, Dataset

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


from src_port.prompt_templates import (
    pseudo_name_mapping, 
    pseudo_name_instr_mapping, 
    PROMPT_TEMPLATES
)

from src_port.utils import (
    compute_similarity,
    compute_embeddings,
    get_gradient_norm,
    encode_query,
    embed_corpus,
    get_ndcg_scores
)

from src_port.dataset_helper import (
    DatasetDownloader,
    get_eval_dataloader,
    get_train_dataloader
)

from src_port.loss_functions import (
    compute_perplexity,
    get_batch_logps,
    odds_ratio_loss,
    compute_Pr,
    get_perplexity,
    compute_loss
)


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

    with torch.no_grad():
        for bid, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            #gold_ids = batch["gold_retrieval_ids"].to(device)
            #print(f"MAX IDS: {torch.max(gold_ids,-1).values.max(-1).values.item()} | Corpus shape: {corpus_embeddings.shape}")
            queries = batch["query"]
            bs = queries["input_ids"].shape[0]


            gold_ids = []
            for _i, i in enumerate(range(bid*bs,bid*bs+bs)):
                pos = eval_triplets[i]["positive"]
                # print("-"*100)
                # print(pos)
                # print("-"*30)
                # print(eval_triplets[i]["query"])
                # print("-"*50)
                # print(retr_tokenizer.decode(batch["positive"]["input_ids"][_i,:], remove_special_tokens=True))
                # print("-"*30)
                # print(retr_tokenizer.decode(batch["query"]["input_ids"][_i,:], remove_special_tokens=True))
                # print("-"*100)
                idx = api_corpus.index(pos)
                gold_ids.append(idx)
            gold_ids = torch.tensor(gold_ids).view(1,-1).to(device)            

            # Compute query embeddings
            query_embeddings = encode_query(retr_model, queries, device)

            # Compute similarities with the entire corpus
            all_similarities = torch.matmul(query_embeddings, corpus_embeddings.T)  # [bs, num_docs]
            all_similarities = all_similarities.squeeze(0)

            # Compute ranks
            _, indices = all_similarities.topk(k, dim=1, largest=True)


            for _k in range(k):
              #rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.unsqueeze(0).T, dim=-1).sum()
              rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.T, dim=-1).sum()
              ranks[_k] += rank_at_n
            
            # Compute NDCG score
            this_ndcg_scores = get_ndcg_scores(ndcg_k_values, batch, gold_ids.squeeze(0), all_similarities, api_corpus, [eval_triplets[i] for i in range(bid*bs, bid*bs+bs)])
            for i, _ in enumerate(this_ndcg_scores):
                ndcg_scores[i] += this_ndcg_scores[i]

    # Normalize ranks
    num_samples = len(eval_dataloader.dataset)
    ranks = [r / num_samples * 100 for r in ranks]
    
    ndcg_score_avg = [0 for _ in range(len(ndcg_k_values))]
    for i, k_val in enumerate(ndcg_k_values):
        ndcg_score_avg[i] = sum(ndcg_scores[i]) / num_samples

    return ranks, ndcg_score_avg



def run_evaluation(retr_model : AutoModel,
                   retr_tokenizer : AutoTokenizer,
                   dataset : Dataset,
                   eval_api_corpus : List[str],
                   retriever_max_seq_length : int,
                   eval_batch_size : int,
                   preprocessing_batch_size : int,
                   device : str,
                   k_eval : int):
    retr_model.eval()

    logger.info("Get Eval DataLoader")
    eval_data_config = {
        "dataset" : dataset, 
        "api_corpus_list" : eval_api_corpus,
        "retrieval_max_length" : retriever_max_seq_length,
        "retrieval_tokenizer" : retr_tokenizer,
        "batch_size" : eval_batch_size
    }
    eval_triplet_dataloader, eval_triplets = get_eval_dataloader(**eval_data_config)

    logger.info("Embedding Tool Corpus")
    eval_corpus_embeddings = embed_corpus(retr_model,
                                        retr_tokenizer,
                                        eval_api_corpus,
                                        device,
                                        batch_size=preprocessing_batch_size,
                                        max_length=retriever_max_seq_length)
    eval_corpus_embeddings = eval_corpus_embeddings.to(device)

    logger.info("Compuring rank accuracy")
    ranks, ndcg_scores = evaluate(retr_model,
                    eval_triplet_dataloader,
                    eval_triplets,
                    eval_corpus_embeddings,
                    device,
                    k=k_eval,
                    api_corpus=eval_api_corpus,
                    retr_tokenizer=retr_tokenizer)

    # Print results
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

    log_eval = {}
    for n in range(k_eval):
        log_eval[f"RANK@{n+1}"] = ranks[n]

    for n, _k in enumerate(ndcg_k_values):
        log_eval[f"NDCG@{_k}"] = ndcg_scores[n]

    wandb.log(log_eval)

    retr_model.train()
    del eval_corpus_embeddings
    torch.cuda.empty_cache()


def train(dataset : Dataset,
          dataset_name : str,
          retr_tokenizer : AutoTokenizer, 
          retr_model : AutoModel,
          infer_tokenizer : AutoTokenizer,
          infer_model : AutoModelForCausalLM,
          train_api_corpus : List[str],
          eval_api_corpus : List[str],
          data_collator_completion : DataCollatorMixin,
          eval_strategy : str = "epoch",
          eval_steps : int = None,
          prompt_template : str = "",
          instruction_prompt : str = "",
          lambda_loss : float = 0.2,
          beta : float = 1,
          gamma : float = 1,
          num_epochs : int = 10,
          learning_rate : float = 1e-4,
          scheduler_type : str = "cosine",
          retriever_max_seq_length : int = 514,
          inference_max_seq_length : int = 1024,
          number_of_neg_examples : int = 3,
          train_batch_size : int = 2,
          eval_batch_size : int = 2,
          preprocessing_batch_size : int = 4,
          log_freq :  int = 100,
          k_eval : int = 10,
          device : str = "cuda",
          wandb_project_name : str = "",
          wandb_run_name : str = ""):



    # Initialize wandb
    config = {
        "beta" : beta,
        "gamma" : gamma,
        "lr" : learning_rate,
        "scheduler_type" : scheduler_type,
        "train_batch_size" : train_batch_size,
        "eval_batch_size" : eval_batch_size,
        "lambda_loss_factor" : lambda_loss,
        "retriever_max_seq_length" : retriever_max_seq_length,
        "inference_max_seq_length" : inference_max_seq_length,
        "epochs" : num_epochs,
        "dataset" : dataset_name
    }

    wandb.init(project=wandb_project_name, 
               name=wandb_run_name,
               config=config)
    wandb.watch(retr_model, log_freq=log_freq)
    
    # First evaluation
    logger.info(f"Starting Initial Evaluation")
    eval_config = {
        "retr_model" : retr_model,
        "retr_tokenizer" : retr_tokenizer,
        "dataset" : dataset,
        "eval_api_corpus" : eval_api_corpus,
        "retriever_max_seq_length" : retriever_max_seq_length,
        "eval_batch_size" : eval_batch_size,
        "preprocessing_batch_size" : preprocessing_batch_size,
        "device" : device,
        "k_eval" : k_eval
    }
    run_evaluation(**eval_config)
    


    # Load first dataloader
    train_data_config = {
        "dataset" : dataset,
        "api_corpus_list" : train_api_corpus,
        "retrieval_max_length" : retriever_max_seq_length,
        "generateor_max_length" : inference_max_seq_length,
        "retrieval_tokenizer" : retr_tokenizer,
        "inference_tokenizer" : infer_tokenizer,
        "epoch_number" : 0,
        "batch_size" : train_batch_size,
        "prompt_template" : prompt_template,
        "num_neg_examples" : number_of_neg_examples
    }
    triplet_dataloader = get_train_dataloader(**train_data_config)

    optimizer = torch.optim.AdamW(retr_model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(triplet_dataloader)
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    kl_div = KLDivLoss(reduction='none')

    for epoch in range(num_epochs):
        retr_model.train()

        logger.info("Creating pseudo-random dataloader")
        train_data_config = {
            "dataset" : dataset,
            "api_corpus_list" : train_api_corpus,
            "retrieval_max_length" : retriever_max_seq_length,
            "generateor_max_length" : inference_max_seq_length,
            "retrieval_tokenizer" : retr_tokenizer,
            "inference_tokenizer" : infer_tokenizer,
            "epoch_number" : epoch,
            "batch_size" : train_batch_size,
            "prompt_template" : prompt_template,
            "num_neg_examples" : number_of_neg_examples
        }
        triplet_dataloader = get_train_dataloader(**train_data_config)

        logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")

        pbar = tqdm(enumerate(triplet_dataloader), total=len(triplet_dataloader), desc="Training PORT with RePlug+ORPO")
        for bid, batch in pbar:

            bs = batch["query"]["input_ids"].shape[0]

            pos_labels = batch["labels_pos"]    # [bs, max_seq_len]
            neg_labels = batch["labels_neg"]    # bs * [n_neg_docs, max_seq_len]

            queries = batch["query"]            # [bs, max_seq_len]
            pos_docs = batch["positive"]        # [bs, max_seq_len]
            neg_docs =  batch["negative"]       # bs * [n_neg_docs, max_seq_len]


            n_neg_docs = neg_docs[0]["input_ids"].shape[0]

            # > Compute positive-negative similarities
            pos_simialrity = compute_similarity(retr_model, queries, pos_docs).view(bs,-1)  # [bs,1]


            neg_similarity = []
            for nid in range(n_neg_docs):
                # consider the i-th negative document from each batch
                data = {
                    k : torch.stack([neg_docs[bid][k][nid,:] for bid in range(bs)]) for k in ['input_ids', 'attention_mask']
                }
                this_neg_similarity = compute_similarity(retr_model, 
                                                        queries, 
                                                        data).view(bs,-1)
                neg_similarity.append(this_neg_similarity)
            

            neg_similarity = torch.stack(neg_similarity, dim=-1)
            similarities = torch.cat((pos_simialrity, neg_similarity.squeeze(1)), dim=-1) # [bs, 1+n_neg_docs]


            # Normalize and weight similarities into retrieval probabilities
            similarities = similarities / gamma # weight probabilities with the set hyperparameter

            Pr_retr = compute_Pr(
                similarities = similarities,
                axis = -1
            )

            # > Get prompts
            input_prompt_pos = batch["q_pos_prompt"]
            input_prompt_neg = batch["q_neg_prompt"]

            # > Inference on the model
            input_prompt_pos = {k : input_prompt_pos[k].to(device) for k in input_prompt_pos}
            input_prompt_neg = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in input_prompt_neg]

            pos_labels = {k : pos_labels[k].to(device) for k in pos_labels}
            neg_labels = [{k : neg_docs_trip[k].to(device) for k in neg_docs_trip} for neg_docs_trip in neg_labels]

            pos_perplexity = []
            neg_perplexity = []

            with torch.no_grad():
                # Positive
                pos_data = next(iter(DataLoader(
                    Dataset.from_dict(input_prompt_pos), 
                    shuffle=False, 
                    batch_size=bs, 
                    collate_fn=data_collator_completion)))

                pos_data = {k: v.to(device) for k, v in pos_data.items()}
                #pos_data["labels"] = pos_labels["input_ids"]
                labels = pos_data.pop("labels")

                outputs_pos = infer_model(**pos_data)

                pos_perplexity = get_perplexity(outputs=outputs_pos, 
                                                input_ids=labels,#pos_data["input_ids"], 
                                                attention_mask=pos_data["attention_mask"],
                                                padding_token_ids=retr_tokenizer.pad_token_id)
                del outputs_pos, pos_data
                torch.cuda.empty_cache()

                # Negatives
                for n_id in range(n_neg_docs):
                    neg_data = next(iter(DataLoader(
                        Dataset.from_dict(
                            {k: torch.stack([input_prompt_neg[bid][k][n_id,:] for bid in range(bs)]) 
                            for k in ["input_ids", "attention_mask"]}), 
                        shuffle=False, 
                        batch_size=bs, 
                        collate_fn=data_collator_completion)))

                    neg_data = {k: v.to(device) for k, v in neg_data.items()}
                    #neg_data["labels"] = torch.stack([neg_labels[bid]["input_ids"][n_id,:] for bid in range(bs)], dim=0)
                    #neg_data["labels"] = pos_labels["input_ids"].to(device)
                    labels = neg_data.pop("labels")
                    
                    outputs_neg = infer_model(**neg_data)

                    neg_perplexity.append(get_perplexity(outputs=outputs_neg, 
                                                         input_ids=labels,#neg_data["labels"],#neg_data["input_ids"],
                                                         attention_mask=neg_data["attention_mask"],
                                                         padding_token_ids=retr_tokenizer.pad_token_id))

                    del outputs_neg, neg_data
                    torch.cuda.empty_cache()

            # Compute Q
            neg_perplexity = torch.stack(neg_perplexity, dim=-1)
            concat_perplexities = torch.cat((pos_perplexity.unsqueeze(0).T, neg_perplexity), dim=-1)
            concat_perplexities = concat_perplexities / beta # weight perplexities with the set hyperparameter
            
            Q = F.softmax(concat_perplexities, dim=-1)

            del concat_perplexities
            torch.cuda.empty_cache()

            # >> Compute losses

            # - Probability-Perplexit Odds Ratio
            ppl_pr_KL_loss = compute_loss(Q, Pr_retr, kl_div)

            # - Odds Preference Alignment
            pref_loss = 0
            pos_rewards, neg_rewards = [], []
            pref_ratio, maean_prob_ratio = 0, 0

            pos_retrieval_prob = Pr_retr[:, 0]
            neg_retrieval_probs = Pr_retr[:, 1:]

            # for each pair (pos, neg_i)
            for neg_i in range(n_neg_docs):
                neg_retrieval_prob = neg_retrieval_probs[:, neg_i]
                
                _pref_loss, _pos_reward, _neg_reward, _pref_ratio, _maean_prob_ratio = odds_ratio_loss(
                    positive_retr_log_prob=pos_retrieval_prob.log(),
                    negative_retr_log_prob=neg_retrieval_prob.log(),
                    beta=1#beta
                )
                pref_loss += _pref_loss.mean()
                pos_rewards.append(_pos_reward.mean())  # weighted log retr prob (w = beta) for the positive sample
                neg_rewards.append(_neg_reward.mean())  # weighted log retr prob (w = beta) for the negative sample
                pref_ratio += _pref_ratio               # average unweighed pref loss
                maean_prob_ratio += _maean_prob_ratio      # average log ration

            pref_loss /= n_neg_docs
            pref_ratio /= n_neg_docs
            maean_prob_ratio /= n_neg_docs

            # > Aggregate loss
            loss = ppl_pr_KL_loss - lambda_loss * pref_loss

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(retr_model.parameters(), max_norm=1.0) # apply gradient clipping
            optimizer.step()
            lr_scheduler.step()
            

            # Compute gradient norm
            grad_norm = get_gradient_norm(retr_model)

            optimizer.zero_grad()
            

            # how many times the probability of retrieving a positive overcome the one of retrieving a negative
            pos_rewards = torch.tensor(pos_rewards)
            neg_rewards = torch.tensor(neg_rewards)
            retrieval_accuracy = (pos_rewards > neg_rewards).float()

            pbar.set_postfix({'Loss': loss.item(), 'RePlug Loss' : ppl_pr_KL_loss.item(), "OPRO Loss" : -pref_loss.item(), "Retrieval Compared Accuracy" : retrieval_accuracy.mean().cpu().item()})


            if bid % log_freq == 0:
                
                wandb.log({
                    "ports_loss": loss.item(),
                    "replug_loss": ppl_pr_KL_loss.item(),
                    "odds_ratio_loss": -pref_loss.item(),
                    "ratio_reward" : pref_ratio,
                    "retrieval_accuracy" : retrieval_accuracy.mean().cpu(), 
                    "positive_retrieval_probability" : pos_retrieval_prob.mean().cpu(),
                    "positive_log_probability" : pos_rewards.mean().cpu(),
                    "negative_retrieval_probability" : neg_retrieval_probs.mean().cpu(),
                    "negative_log_probability" : neg_rewards.mean().cpu(),
                    "positive_ppl" : pos_perplexity.mean().cpu(),
                    "positive_Q_ppl" : Q[:,0].mean().cpu(),
                    "positive_sim" : pos_simialrity.mean().cpu(),
                    "negative_sim" : neg_similarity.mean(-1).mean().cpu(),
                    "negative_ppl" : neg_perplexity.mean(-1).mean().cpu(),
                    "negative_Q_ppl" : Q[:,1:].mean(-1).mean().cpu(),
                    "mean_retr_prob_ration" : maean_prob_ratio,
                    "gradient_norm" : grad_norm,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch" : epoch + 1
                })

            del Q, Pr_retr, ppl_pr_KL_loss, pref_loss, loss, pos_retrieval_prob, neg_retrieval_probs, neg_retrieval_prob
            del neg_perplexity, pos_perplexity
            del pref_ratio, retrieval_accuracy, pos_rewards, neg_rewards, maean_prob_ratio
            torch.cuda.empty_cache()

            if eval_strategy == "steps" and bid % eval_steps == 0 and bid != 0:

                logger.info(f"Starting evaluation epoch {epoch+1}/{num_epochs} - step {bid*eval_steps}")
                eval_config = {
                    "retr_model" : retr_model,
                    "retr_tokenizer" : retr_tokenizer,
                    "dataset" : dataset,
                    "eval_api_corpus" : eval_api_corpus,
                    "retriever_max_seq_length" : retriever_max_seq_length,
                    "eval_batch_size" : eval_batch_size,
                    "preprocessing_batch_size" : preprocessing_batch_size,
                    "device" : device,
                    "k_eval" : k_eval
                }
                run_evaluation(**eval_config)

        if eval_strategy == "epoch":
            logger.info(f"Starting evaluation epoch {epoch+1}/{num_epochs}")
            eval_config = {
                "retr_model" : retr_model,
                "retr_tokenizer" : retr_tokenizer,
                "dataset" : dataset,
                "eval_api_corpus" : eval_api_corpus,
                "retriever_max_seq_length" : retriever_max_seq_length,
                "eval_batch_size" : eval_batch_size,
                "preprocessing_batch_size" : preprocessing_batch_size,
                "device" : device,
                "k_eval" : k_eval
            }
            run_evaluation(**eval_config)

    logger.info("Training and evaluations are over")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='PORT training')
    parser.add_argument('--dataset', type=str, default="bfcl", choices=["bfcl", "apibank", "apibench", "octopus", "octopus-overlap", "toole", "toole-overlap", "toolbench"], help='Dataset name for training and avaluation')

    # Models
    parser.add_argument('--inference_model_name', type=str, default="llama3-8B", choices=["llama3-8B", "codestral-22B", "gemma2-2B", "groqLlama3Tool-8B"], help="Pseudo-Name of the generative model to use for function calling")
    parser.add_argument('--retrieval_model_name', type=str, default="FacebookAI/roberta-base", help="Name of the encoder model to use for retrieval")
    parser.add_argument('--retriever_max_seq_length', type=int, default=514, help="Max sequence length for retriever")
    parser.add_argument('--inference_max_seq_length', type=int, default=1024, help="Max sequence length for the inference model")

    # General
    parser.add_argument('--do_train', action='store_true', default=False, help="Whether to run the training loop")
    parser.add_argument('--do_eval', action='store_true', default=False,  help="Whether to run the evaluation loop")
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help="Whether to load the model in 4 bit")

    parser.add_argument('--eval_strategy', type=str, default="epoch", choices=["epoch", "steps"], help="Strategy to use for evaluation")
    parser.add_argument('--eval_steps', type=int, default=None, help="Number of steps after which the evaluation is performed if eval_strategy = 'steps'")

    # Training/Eval config
    parser.add_argument('--max_train_samples', type=int, default=None, help="Maximum number of training instances to retain (all if set to None)")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Maximum number of evaluation instances to retain (all if set to None)")

    parser.add_argument('--n_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_type', type=str, default="cosing", help="Learning rate scheduler approach")
    parser.add_argument('--train_batch_size', type=int,default=2, help="Batch size for training")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="Batch size for evaluation")
    parser.add_argument('--preprocessing_batch_size', type=int, default=4, help="Batch size for the preprocessing phase")
    parser.add_argument('--padding_side', type=str, default="right", help="Padding side for tokenizers")

    # Additional arguments based on the training and evaluation loop
    parser.add_argument('--lambda_loss', type=float, default=0.2, help="Lambda weighting factor parameter")
    
    parser.add_argument('--n_neg_examples', type=int, default=3, help="Number of negative samples to include in the triplets")
    parser.add_argument('--k_eval', type=int, default=10, help="Number of R@K value to test during evaluation")
    parser.add_argument('--gamma', type=float, default=1, help="Gamma parameter for computing Pr_retr")
    parser.add_argument('--beta', type=float, default=1, help="Beta parameter for softmax in Q computation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    parser.add_argument('--wandb_project_name', type=str, default="PortsAAAI", help="WandbB project name")
    parser.add_argument('--wandb_run_name', type=str, default="test_run", help="WandbB run name")

    parser.add_argument('--log_freq', type=int, default=100, help="Logging frequency")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # > Load models
    logger.info("Loading Retrieval Model")
    retr_model_name = args.retrieval_model_name
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model = AutoModel.from_pretrained(retr_model_name).to(device)
    

    logger.info("Loading Generative Model")
    pseudo_model_name = args.inference_model_name
    infer_model_name = pseudo_name_mapping[pseudo_model_name]
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    if args.load_in_4bit:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, 
                                                           quantization_config=nf4_config, 
                                                           device_map=0 if device=="cuda" else "cpu",
                                                           attn_implementation="flash_attention_2")
    else:
        infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, 
                                                           device_map=0 if device=="cuda" else "cpu",
                                                           attn_implementation="flash_attention_2")

    
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
        infer_tokenizer.pad_token_id = infer_tokenizer.eos_token_id

        infer_model.resize_token_embeddings(len(infer_tokenizer))

    infer_tokenizer.padding_side=args.padding_side

    if retr_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        retr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        retr_tokenizer.pad_token = retr_tokenizer.eos_token
        retr_tokenizer.pad_token_id = retr_tokenizer.eos_token_id

        retr_model.resize_token_embeddings(len(retr_tokenizer))

    retr_tokenizer.padding_side=args.padding_side


    # > Load dataset
    logger.info("Loading dataset")
    dataset_downloader = DatasetDownloader(dataset_name=args.dataset)
    dataset = dataset_downloader.get_dataset()
    dataset = dataset_downloader.post_process_answers(dataset)

    # Sample from dataset if necessary
    if args.max_train_samples:
        n_inst = min(args.max_train_samples, len(dataset["train"]))
        selected_indices = random.sample(range(len(dataset["train"])), n_inst)
        dataset["train"] = dataset["train"].select(selected_indices)

    if args.max_eval_samples:
        n_inst = min(args.max_eval_samples, len(dataset["test"]))
        selected_indices = random.sample(range(len(dataset["test"])), n_inst)
        dataset["test"] = dataset["test"].select(selected_indices)

    # > Create and embed corpora of functions
    logger.info("Defining tool corpora")
    
    # TODO: add if do_train ...
    train_api_corpus = list(set(dataset["train"]["api_description"]))
    eval_api_corpus = list(set(dataset["test"]["api_description"]))

    logger.info(f"[STAST] Corpus--> Train: {len(train_api_corpus)} | Eval: {len(eval_api_corpus)}")

    k_eval = min(args.k_eval,len(eval_api_corpus))

    logger.info("Setting the prompt and answer templates")
    prompt_template = PROMPT_TEMPLATES[pseudo_model_name]
    instruction = pseudo_name_instr_mapping[pseudo_model_name]
    answer_template = prompt_template["answer_template"]

    response_template_ids = infer_tokenizer.encode(answer_template,
                                                    add_special_tokens=False)#[2:]

    data_collator_completion = DataCollatorForCompletionOnlyLM(tokenizer=infer_tokenizer,
                                                                response_template=response_template_ids,
                                                                mlm=False)


    # Start training/evaluation

    train_eval_config = {
        "dataset" : dataset,
        "dataset_name" : args.dataset,
        "retr_tokenizer" : retr_tokenizer, 
        "retr_model" : retr_model,
        "infer_tokenizer" : infer_tokenizer,
        "infer_model" : infer_model,
        "train_api_corpus" : train_api_corpus,
        "eval_api_corpus" : eval_api_corpus,
        "eval_strategy" : args.eval_strategy,
        "eval_steps" : args.eval_steps,
        "prompt_template" : prompt_template["prompt_template"],
        "instruction_prompt" : instruction,
        "data_collator_completion" : data_collator_completion,
        "lambda_loss" : args.lambda_loss,
        "beta" : args.beta,
        "gamma" : args.gamma,
        "num_epochs" : args.n_epochs,
        "retriever_max_seq_length" : args.retriever_max_seq_length,
        "inference_max_seq_length" : args.inference_max_seq_length,
        "number_of_neg_examples" : args.n_neg_examples,
        "train_batch_size" : args.train_batch_size,
        "eval_batch_size" : args.eval_batch_size,
        "preprocessing_batch_size" : args.preprocessing_batch_size,
        "learning_rate" : args.lr,
        "scheduler_type" : args.lr_type,
        "log_freq" :  args.log_freq,
        "k_eval" : k_eval,
        "device" : device,
        "wandb_project_name" : args.wandb_project_name,
        "wandb_run_name" : args.wandb_run_name
    }
    logger.info("Starting Training and Evaluation")
    train(**train_eval_config)
    logger.info("Success. Exit.")


if __name__ == "__main__":
    main()