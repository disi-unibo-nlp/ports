import wandb
import math
from tqdm import tqdm
import argparse
import random

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    set_seed
)
import torch

import logging

formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logging.getLogger().handlers[0].setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROJECT_NAME = "TEST_Port"
RUN_NAME = "PORT_pos_neg_labels"


from src_port.prompt_template import pseudo_name_mapping, pseudo_name_instr_mapping, PROMPT_TEMPLATES

import compute_similarity, compute_Pr, get_perplexity, get_batch_logps, compute_loss_replug, odds_ratio_loss, embed_corpus


from dataset_helper import DatasetDownloader

def evaluate(retr_model,
             eval_dataloader,
             corpus_embeddings,
             device, k):
    retr_model.eval()

    ranks = [0 for _ in range(k)]

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            gold_ids = batch["gold_retrieval_ids"].to(device)
            queries = batch["query"]

            # Compute query embeddings
            query_embeddings = encode_query(retr_model, queries, device)

            # Compute similarities with the entire corpus
            all_similarities = torch.matmul(query_embeddings, corpus_embeddings.T)  # [bs, num_docs]
            all_similarities = all_similarities.squeeze(0)

            # Compute ranks
            _, indices = all_similarities.topk(k, dim=1, largest=True)

            for _k in range(k):
              rank_at_n = torch.any(indices[:, :_k+1] == gold_ids.T, dim=-1).sum()
              ranks[_k] += rank_at_n

    # Normalize ranks
    num_samples = len(eval_dataloader.dataset)
    ranks = [r / num_samples * 100 for r in ranks]

    return ranks

def old_train(dataset : Dataset,
          retr_tokenizer : AutoTokenizer, 
          retr_model : AutoModel,
          infer_tokenizer : AutoTokenizer,
          infer_model : AutoModelForCausalLM,
          train_api_corpus : List[str],
          eval_api_corpus : List[str],
          lambda_loss : float = 0.2,
          train_batch_size : int = 2,
          eval_batch_size : int = 2,
          num_epochs : int = 10,
          retriever_max_seq_length : int = 514,
          inference_max_seq_length : int = 1024,
          number_of_neg_examples : int = 3,
          log_freq :  int = 100,
          k_eval : int = 10,
          device : str = "cuda"):
    
    # Initialize wandb
    wandb.init(project=PROJECT_NAME, 
               name=RUN_NAME)


    retr_model.train()
    for epoch in range(num_epochs):

        logger.info("Creating pseudo-random dataloader")
        train_data_config = {
            "api_corpus_list" : train_api_corpus,
            "retrieval_max_length" : retriever_max_seq_length,
            "generateor_max_length" : inference_max_seq_length,
            "retrieval_tokenizer" : retr_tokenizer,
            "inference_tokenizer" : infer_tokenizer,
            "epoch_number" : epoch,
            "batch_size" : train_batch_size,
            "num_neg_examples" : number_of_neg_examples
        }
        triplet_dataloader = get_train_dataloader(**train_data_config)

        logger.info(f"Starting training epoch {epoch+1}/{num_epochs}")

        pbar = tqdm(enumerate(triplet_dataloader), total=len(triplet_dataloader), desc="Training PORT with RePlug+ORPO")
        for bid, batch in pbar:

            bs = batch["query"]["input_ids"].shape[0]
            n_neg_docs = neg_docs[0]["input_ids"].shape[0]

            pos_labels = batch["labels_pos"]    # [bs, max_seq_len]
            neg_labels = batch["labels_neg"]    # bs * [n_neg_docs, max_seq_len]

            queries = batch["query"]            # [bs, max_seq_len]
            pos_docs = batch["positive"]        # [bs, max_seq_len]
            neg_docs =  batch["negative"]       # bs * [n_neg_docs, max_seq_len]


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
            Pr_retr = compute_Pr(
                similarities = similarities,
                gamma = gamma,
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

            # > Define dataloader with response_template masking
            pos_inner_data_loader = DataLoader(
                Dataset.from_dict(input_prompt_pos),
                shuffle=False,
                batch_size=bs,
                collate_fn=data_collator_completion
            )

            neg_inner_data_loader = [DataLoader(
                Dataset.from_dict({k : torch.stack([input_prompt_neg[bid][k][nid,:] for bid in range(bs)]) for k in ["input_ids", "attention_mask"]}),
                shuffle=False,
                batch_size=bs,
                collate_fn=data_collator_completion
            ) for nid in range(n_neg_docs)]

            pos_perplexity = []
            neg_perplexity = []

            with torch.no_grad():

                # >> RUN FORWARD

                # > POSITIVE
                # - move data to device
                _,pos_data = next(enumerate(pos_inner_data_loader))
                pos_data = {k:pos_data[k].to(device) for k in pos_data}
                pos_data["labels"] = pos_labels["input_ids"].to(device)

                # - forward
                outputs_pos = infer_model(**pos_data)

                # - compute perplexity
                pos_ppl_config = {
                    #"loss" : outputs_pos.loss,
                    "outputs" : outputs_pos,
                    "input_ids" : pos_data["input_ids"],
                    "attention_mask" : pos_data["attention_mask"]
                }
                pos_perplexity = get_perplexity(**pos_ppl_config)


                # >> COMPUTE LOG-PROBABILITY FOR ORPO
                pos_logps = get_batch_logps(logits=outputs_pos.logits,
                                            labels=pos_data["labels"], # masked with response template
                                            average_log_prob=True,
                                            label_pad_token_id=infer_tokenizer.pad_token_id)

                del outputs_pos
                torch.cuda.empty_cache()

                # > NEGATIVES
                neg_data_out = []
                neg_logps = []

                for n_id in range(n_neg_docs):

                    # - compose data and move to device
                    _,neg_data = next(enumerate(neg_inner_data_loader[n_id]))
                    neg_data = {k:neg_data[k].to(device) for k in neg_data}
                    _neg_labels = torch.stack([neg_labels[bid]["input_ids"][n_id,:] for bid in range(bs)], dim=0)
                    neg_data["labels"] = _neg_labels.to(device)

                    # - forward
                    outputs_neg = infer_model(**neg_data)
                    neg_data_out.append(outputs_neg)

                    # - compute perplexity
                    neg_ppl_config = {
                        "outputs" : outputs_neg,
                        "input_ids" : neg_data["input_ids"],
                        "attention_mask" : neg_data["attention_mask"]
                    }
                    _neg_perplexity = get_perplexity(**neg_ppl_config)
                    neg_perplexity.append(_neg_perplexity)

                    # >> COMPUTE LOG-PROBABILITY FOR ORPO
                    this_neg_logps = get_batch_logps(logits=outputs_neg.logits,
                                                    labels=neg_data["labels"],
                                                    average_log_prob=True,
                                                    label_pad_token_id=infer_tokenizer.pad_token_id)
                    neg_logps.append(this_neg_logps)

                    del outputs_neg
                    torch.cuda.empty_cache()

            del pos_inner_data_loader, neg_inner_data_loader
            torch.cuda.empty_cache()


            # >> Compute RePLUG Loss
            neg_perplexity = torch.stack(neg_perplexity, dim=-1)
            concat_perplexities = torch.cat((pos_perplexity.unsqueeze(0).T, neg_perplexity), dim=-1)


            Q = F.softmax(concat_perplexities / beta, dim=-1) # [bs,1+n_docs]

            del neg_perplexity, pos_perplexity
            torch.cuda.empty_cache()

            replug_loss = compute_loss_replug(Q, Pr_retr, kl_div)

            # >> Compute ORPO Loss
            orpo_loss, pref_ratio, log_odds_chosen = 0,0,0
            pos_rewards, neg_rewards = [],[]

            # - for each pair (pos, neg_i)
            for neg_i_logps in neg_logps:
                _orpo_loss, _pos_reward, _neg_reward, _pref_ratio, _log_odds_chosen = odds_ratio_loss(
                    policy_chosen_logps=pos_logps,
                    policy_rejected_logps=neg_i_logps,
                    beta=0.1
                )
                orpo_loss += _orpo_loss.mean()
                pos_rewards.append(_pos_reward.mean())
                neg_rewards.append(_neg_reward.mean())
                pref_ratio += _pref_ratio
                log_odds_chosen += _log_odds_chosen


            orpo_loss = orpo_loss.mean()
            del neg_logps, pos_logps
            torch.cuda.empty_cache()

            loss = replug_loss - lambda_loss * orpo_loss

            if math.isinf(loss):
                logger.info(f"Discarded infinite loss perturbation")
            else:
                loss.backward()

            # Compute gradient norm
            grad_norm = get_gradient_norm(retr_model)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            reward_accuracies = (torch.tensor(pos_rewards) > torch.tensor(neg_rewards)).float()
            pbar.set_postfix({'Loss': loss.item(), 'RePlug Loss' : replug_loss.item(), "OPRO Loss" : -orpo_loss.item(), "Reward Accuracy" : reward_accuracies.mean().cpu().item()})


            if bid % log_freq == 0 or bid == 0:
                
                wandb.log({
                    "port_loss": loss.item(),
                    "replug_loss": replug_loss.item(),
                    "odds_ratio_loss": -orpo_loss.item(),
                    "ratio_reward" : pref_ratio,
                    "reward_Accuracy" : reward_accuracies.mean().cpu(),
                    "positive_reward" : pos_reward.detach().mean().cpu(),
                    "negative_reward" : neg_reward.detach().mean().cpu(),
                    "positive_odds_chosen" : log_odds_chosen,
                    "positive_logpb" : pos_logps.detach().mean().cpu(),
                    "negative_logpb" : neg_logps.detach().mean().cpu(),
                    "gradient_norm" : grad_norm,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch" : epoch + 1
                })

            del Q, Pr_retr # neg_perplexity
            torch.cuda.empty_cache()

        retr_model.eval()
        eval_corpus_embeddings = embed_corpus(retr_model,
                                            eval_api_corpus,
                                            device,
                                            batch_size=eval_batch_size,
                                            max_length=retriever_max_seq_length)
        eval_corpus_embeddings = eval_corpus_embeddings.to(device)

        ranks = evaluate(retr_model,
                        eval_triplet_dataloader,
                        eval_corpus_embeddings,
                        device,
                        k=k_eval)

        # Print results
        print("\n\n")
        print("EVALUATION")
        print("*"*50)
        for n in range(K_EVAL):
            print(f"RANK@{n+1}: {ranks[n]:.2f}%")
        print("*"*50)
        print("\n\n")

        wandb.log({
            f"RANK@{n+1}: {ranks[n]:.2f}%" for n in range(K_EVAL)
        })

        retr_model.train()
        del embedded_documents
        torch.cuda.empty_cache()

    wandb.finish()

def train(dataset : Dataset,
          retr_tokenizer : AutoTokenizer, 
          retr_model : AutoModel,
          infer_tokenizer : AutoTokenizer,
          infer_model : AutoModelForCausalLM,
          train_api_corpus : List[str],
          eval_api_corpus : List[str],
          data_collator_completion : DataCollatorMixin,
          prompt_template : str = "",
          instruction_prompt : str = "",
          lambda_loss : float = 0.2,
          num_epochs : int = 10,
          retriever_max_seq_length : int = 514,
          inference_max_seq_length : int = 1024,
          number_of_neg_examples : int = 3,
          train_batch_size : int = 2,
          eval_batch_size : int = 2,
          preprocessing_batch_size : int = 4,
          log_freq :  int = 100,
          k_eval : int = 10,
          device : str = "cuda"):

    # Initialize wandb
    wandb.init(project=PROJECT_NAME, 
               name=RUN_NAME)


    retr_model.train()
    for epoch in range(num_epochs):

        logger.info("Creating pseudo-random dataloader")
        train_data_config = {
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
            n_neg_docs = neg_docs[0]["input_ids"].shape[0]

            pos_labels = batch["labels_pos"]    # [bs, max_seq_len]
            neg_labels = batch["labels_neg"]    # bs * [n_neg_docs, max_seq_len]

            queries = batch["query"]            # [bs, max_seq_len]
            pos_docs = batch["positive"]        # [bs, max_seq_len]
            neg_docs =  batch["negative"]       # bs * [n_neg_docs, max_seq_len]


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
            Pr_retr = compute_Pr(
                similarities = similarities,
                gamma = gamma,
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
                pos_data["labels"] = pos_labels["input_ids"]

                outputs_pos = infer_model(**pos_data)
                pos_perplexity = get_perplexity(outputs=outputs_pos, 
                                                input_ids=pos_data["input_ids"], 
                                                attention_mask=pos_data["attention_mask"])

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
                    neg_data["labels"] = torch.stack([neg_labels[bid]["input_ids"][n_id,:] for bid in range(bs)], dim=0)

                    outputs_neg = infer_model(**neg_data)
                    neg_perplexity.append(get_perplexity(outputs=outputs_neg, 
                                                         input_ids=neg_data["input_ids"],
                                                         attention_mask=neg_data["attention_mask"]))

                    del outputs_neg, neg_data
                    torch.cuda.empty_cache()

            # Compute Q
            neg_perplexity = torch.stack(neg_perplexity, dim=-1)
            concat_perplexities = torch.cat((pos_perplexity.unsqueeze(0).T, neg_perplexity), dim=-1)
            Q = F.softmax(concat_perplexities / beta, dim=-1)

            del neg_perplexity, pos_perplexity, concat_perplexities
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

            for neg_i in range(n_neg_docs):
                neg_retrieval_prob = neg_retrieval_probs[:, neg_i]
                
                _pref_loss, _pos_reward, _neg_reward, _pref_ratio, _maean_prob_ratio = odds_ratio_loss(
                    positive_retr_log_prob=pos_retrieval_prob.log(),
                    positive_retr_log_prob=neg_retrieval_prob.log(),
                    beta=0.1
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

            if math.isinf(loss):
                logger.info(f"Discarded infinite loss perturbation")
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            # Compute gradient norm
            grad_norm = get_gradient_norm(retr_model)

            optimizer.zero_grad()

            # how many times the probability of retrieving a positive overcome the one of retrieving a negative
            retrieva_accuracy = (torch.tensor(pos_rewards) > torch.tensor(neg_rewards)).float()

            pbar.set_postfix({'Loss': loss.item(), 'RePlug Loss' : replug_loss.item(), "OPRO Loss" : -orpo_loss.item(), "Retrieval Compared Accuracy" : retrieva_accuracy.mean().cpu().item()})


            if bid % log_freq == 0 or bid == 0:
                
                wandb.log({
                    "ports_loss": loss.item(),
                    "replug_loss": ppl_pr_KL_loss.item(),
                    "odds_ratio_loss": -pref_loss.item(),
                    "ratio_reward" : pref_ratio.item(),
                    "retrieval_accuracy" : retrieva_accuracy.mean().cpu(), 
                    "positive_probability" : pos_reward.detach().mean().cpu(),
                    "negative_probability" : neg_reward.detach().mean().cpu()
                    "mean_retr_prob_ration" : maean_prob_ratio,
                    "gradient_norm" : grad_norm,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch" : epoch + 1
                })

            del Q, Pr_retr, ppl_pr_KL_loss, pref_loss, loss, pos_retrieval_prob, neg_retrieval_probs, neg_retrieval_prob
            del pref_ratio, retrieva_accuracy, pos_reward, neg_reward, maean_prob_ratio
            torch.cuda.empty_cache()

        logger.info(f"Starting evaluation epoch {epoch+1}/{n_epochs}")
        retr_model.eval()

        logger.info("Get Eval DataLoader")
        eval_data_config = {
            "dataset" : dataset, 
            "api_corpus_list" : eval_api_corpus,
            "retrieval_max_length" : retriever_max_seq_length,
            "retrieval_tokenizer" : retr_tokenizer,
            "batch_size" : eval_batch_size
        }
        eval_triplet_dataloader = get_eval_dataloader(**eval_data_config)

        logger.info("Embedding Tool Corpus")
        eval_corpus_embeddings = embed_corpus(retr_model,
                                            eval_api_corpus,
                                            device,
                                            batch_size=preprocessing_batch_size,
                                            max_length=retriever_max_seq_length)
        eval_corpus_embeddings = eval_corpus_embeddings.to(device)

        logger.info("Compuring rank accuracy")
        ranks = evaluate(retr_model,
                        eval_triplet_dataloader,
                        eval_corpus_embeddings,
                        device,
                        k=k_eval)

        # Print results
        print("\n\n")
        print("EVALUATION")
        print("*"*50)
        for n in range(K_EVAL):
            print(f"RANK@{n+1}: {ranks[n]:.2f}%")
        print("*"*50)
        print("\n\n")

        wandb.log({
            f"RANK@{n+1}: {ranks[n]:.2f}%" for n in range(K_EVAL)
        })

        retr_model.train()
        del eval_corpus_embeddings
        torch.cuda.empty_cache()

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='PORT training')
    parser.add_argument('--dataset', type=str, default="bfcl", choices=["bfcl", "apibank", "apibench", "octopus", "toole", "toolbench"], help='Dataset name for training and avaluation')

    # Models
    parser.add_argument('--inference_model_name', type=str, default="llama3-8B", choices=["llama3-8B", "codestral-22B", "gemma2-2B", "groqLlama3Tool-8B"], help="Pseudo-Name of the generative model to use for function calling")
    parser.add_argument('--retrieval_model_name', type=str, default="FacebookAI/roberta-base", help="Name of the encoder model to use for retrieval")
    parser.add_argument('--retriever_max_seq_length', type=int, default=514, help="Max sequence length for retriever")
    parser.add_argument('--inference_max_seq_length', type=int, default=1024, help="Max sequence length for the inference model")

    # General
    parser.add_argument('--do_train', action='store_true', default=False, help="Whether to run the training loop")
    parser.add_argument('--do_eval', action='store_true', default=False,  help="Whether to run the evaluation loop")
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help="Whether to load the model in 4 bit")

    # Training/Eval config
    parser.add_argument('--max_train_samples', type=int, default=None, help="Maximum number of training instances to retain (all if set to None)")
    parser.add_argument('--max_eval_samples', type=int, default=None, help="Maximum number of evaluation instances to retain (all if set to None)")

    parser.add_argument('--n_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--lr_type', type=str, default="cosing", help="Learning rate scheduler approach")
    parser.add_argument('--train_batch_size', type=int, help="Batch size for training")
    parser.add_argument('--eval_batch_size', type=int, help="Batch size for evaluation")
    parser.add_argument('--preprocessing_batch_size', type=int, help="Batch size for the preprocessing phase")
    parser.add_argument('--padding_side', type=str, default="right", help="Padding side for tokenizers")

    # Additional arguments based on the training and evaluation loop
    parser.add_argument('--lambda_loss', type=float, default=0.2, help="Lambda weighting factor parameter")
    
    parser.add_argument('--n_neg_examples', type=float, default=3, help="Number of negative samples to include in the triplets")
    parser.add_argument('--k_eval', type=int, default=10, help="Number of R@K value to test during evaluation")
    parser.add_argument('--gamma', type=float, help="Gamma parameter for computing Pr_retr")
    parser.add_argument('--beta', type=float, help="Beta parameter for softmax in Q computation")
    parser.add_argument('--seed', type=float, default=42, help="Random seed")
    

    parser.add_argument('--log_freq', type=int, default=100, help="Logging frequency")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # > Load models
    logger.info("Loading Retrieval Model")
    
    retr_model = AutoModel.from_pretrained(retr_model_name).to(device)
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)

    logger.info("Loading Generative Model")
    pseudo_model_name = args.inference_model_name
    infer_model_name = pseudo_name_mapping[pseudo_model_name]

    infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, 
                                                      load_in_4bit=args.load_in_4bit, 
                                                      device_map=0 if device=="cuda" else "cpu")
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    
    
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        infer_tokenizer.pad_token = infer_tokenizer.eos_token

        infer_model.resize_token_embeddings(len(infer_tokenizer))

    infer_tokenizer.padding_side=args.padding_side

    if retr_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        retr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        retr_tokenizer.pad_token = retr_tokenizer.eos_token

        retr_model.resize_token_embeddings(len(retr_tokenizer))

    retr_tokenizer.padding_side=args.padding_side


    # > Load dataset
    logger.info("Loading dataset")
    dataset_downloader = DatasetDownloader(dataset_name=args.dataset_name)
    dataset = dataset_downloader.get_dataset()
    logger.info("Parsing dataset")
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


    # TODO: answer_template = get_answer_template(gen_model_name)
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
        "retr_tokenizer" : retr_tokenizer, 
        "retr_model" : retr_model,
        "infer_tokenizer" : infer_tokenizer,
        "infer_model" : infer_model,
        "train_api_corpus" : train_api_corpus,
        "eval_api_corpus" : eval_api_corpus,
        "prompt_template" : prompt_template["prompt_template"],
        "instruction_prompt" : instruction,
        "data_collator_completion" : data_collator_completion,
        "lambda_loss" : args.lambda_loss,
        "num_epochs" : args.n_epochs,
        "retriever_max_seq_length" : args.retriever_max_seq_length,
        "inference_max_seq_length" : args.inference_max_seq_length,
        "number_of_neg_examples" : args.n_neg_examples,
        "train_batch_size" : args.train_batch_size,
        "eval_batch_size" : args.eval_batch_size,
        "preprocessing_batch_size" : args.preprocessing_batch_size,
        "log_freq" :  args.log_freq,
        "k_eval" : args.k_eval,
        "device" : device
    }
    logger.info("Starting Training and Evaluation")
    train(**train_eval_config)
    logger.info("Success. Exit.")


if __name__ == '___main__':
    main()