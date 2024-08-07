import random
import json

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from typing import List



# --------------------------------------------------------
# >> TRAINING LOOP HELPERs

def compute_embeddings(model, 
                      documents,
                      device : str = "cuda"):
    """
    Compute the embedding of input documents using the given encoder model.
    """
    # Compute token embeddings
    documents = {k: v.to(device) for k, v in documents.items()}
    model_output = model(**documents)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings_normal = F.normalize(sentence_embeddings, p=2, dim=1)
    del documents, model_output, sentence_embeddings
    return sentence_embeddings_normal


def compute_similarity(model, 
                       queries, 
                       documents,
                       device : str = "cuda"):
    """
    Compute the similarity between queries and documents using the input encoder model.
    """       
    # Compute embedding
    documents_tok = {
        k : documents[k].to(device) for k in documents
    }

    queries_tok = {
        k : queries[k].to(device) for k in queries
    }

    # CLS pooling + normalization
    docs_embeddings = F.normalize(model(**documents_tok)[0][:, 0], p=2, dim=1).unsqueeze(0)
    q_embeddings = F.normalize(model(**queries_tok)[0][:, 0], p=2, dim=1).unsqueeze(0)

    del documents_tok, queries_tok

    cos_sim = F.cosine_similarity(docs_embeddings, q_embeddings, dim=-1)

    return cos_sim


def get_gradient_norm(model):
    """
    Get the current norm of gradients for the input model.
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def encode_query(retr_model, 
                 query,
                 device : str = "cuda"):
    """
    Encode the input query with the given encoder model.
    """
    query = {k:query[k].to(device) for k in query}
    return F.normalize(retr_model(**query)[0][:, 0], p=2, dim=1).unsqueeze(0)


def embed_corpus(retr_model, 
                 retr_tokenizer,
                 corpus, 
                 device, 
                 batch_size=32, 
                 max_length=1024):
    """
    Create embedding matrix for a corpus of documents.
    """
    retr_model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            batch = retr_tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            batch = {k:batch[k].to(device) for k in batch}

            out = retr_model(**batch)

            embeddings = F.normalize(out[0][:, 0], p=2, dim=1)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.squeeze(0)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)