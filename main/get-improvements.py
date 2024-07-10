import os
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    get_scheduler, 
    HfArgumentParser,
    set_seed
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_from_disk
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from tqdm.auto import tqdm
from huggingface_hub import login
from src.data_classes import PyTorchTrainingParams
from src.prompts import PROMPT_TEMPLATES, INSTRUCTIONS
import logging
import wandb

def main():
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('/proj/mounted/improvements.out', mode='w')
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # load models and tokenizers
    set_seed(42)
    retr_tokenizer = AutoTokenizer.from_pretrained("/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/")
    base_model = AutoModel.from_pretrained("/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/").to("cuda")
    finetuned_model = AutoModel.from_pretrained("/proj/mounted/toole_diff_docs_non_overlap").to("cuda")

    base_model.eval()
    finetuned_model.eval()

    def compute_embeddings(model, documents):
        # Compute token embeddings
        documents = {k: v.to("cuda") for k, v in documents.items()}
        model_output = model(**documents)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings_normal = F.normalize(sentence_embeddings, p=2, dim=1)
        del documents, model_output, sentence_embeddings
        return sentence_embeddings_normal

    with open("/proj/mounted/func-docs/documentation-toole-eval.txt", "r") as f:
        func_text = f.read()
    eval_documents = func_text.split("\n")
    tokenized_eval_documents = retr_tokenizer(eval_documents, padding=True, truncation=True, return_tensors='pt')

    def tokenize_function(samples):
        return retr_tokenizer(samples["query"], padding=True, truncation=True, return_tensors='pt')

    dataset = load_from_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")
    dataset = dataset.shuffle(seed=42).flatten_indices()
    input_eval_dataset = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    base_doc_embeddings = compute_embeddings(base_model, tokenized_eval_documents)
    finetuned_doc_embeddings = compute_embeddings(finetuned_model, tokenized_eval_documents)

    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    k = 3
    batch_size = 1
    eval_data_loader = DataLoader(
        input_eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    def get_top_k_docs_per_query(embedded_documents, batch, k, model):
        """
        Compute the top-k documents for each query in the batch, based on their cosine similarity
        """
        embedded_queries = compute_embeddings(model, batch)
        embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
        embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
        cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
        top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
        return top_k_docs.indices, top_k_docs.values

    verify_relevancy = lambda response, doc: response == doc.split(":")[0]

    def get_relevant_docs(response, documents):
        """
        For each example in the batch, retrieve its relevant document
        """        
        for i, doc in enumerate(documents):
                if verify_relevancy(response, doc):
                    return i
        return None
    

    for index, batch in enumerate(eval_data_loader):
        query = dataset["test"]["query"][index]
        response = dataset["test"]["response"][index]
        rel_doc = get_relevant_docs(response, eval_documents)
        docs_per_query_base, sim_per_query_base = get_top_k_docs_per_query(base_doc_embeddings, batch, k, base_model)
        docs_per_query_finetuned, sim_per_query_finetuned = get_top_k_docs_per_query(finetuned_doc_embeddings, batch, k, finetuned_model)
        print(docs_per_query_base)
        print(docs_per_query_finetuned)
        print(rel_doc)
        if ((rel_doc in docs_per_query_finetuned[0][:2] and 
            rel_doc not in docs_per_query_base[0][:2]) or
            (rel_doc == docs_per_query_finetuned[0][0] and 
            rel_doc == docs_per_query_base[0][1])):
            logger.info(query)
        


if __name__ == "__main__":
    main()

