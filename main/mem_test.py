import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModel
login(token=os.getenv('HF_KEY'))
dataset = load_dataset("ToolRetriever/APIBank", "parsed_data")
corpus = list(set(dataset['train']['api_description']))
retr_model_name = "FacebookAI/roberta-base"
retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
retr_model = AutoModel.from_pretrained(retr_model_name).to("cuda")
def embed_corpus(retr_model, corpus, device, batch_size=32, max_length=1024):
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

    stacked_embeddings = torch.stack(all_embeddings)
    return stacked_embeddings.reshape(-1, stacked_embeddings.size(-1))
embed_corpus(retr_model, corpus, "cuda", batch_size=8, max_length=512)