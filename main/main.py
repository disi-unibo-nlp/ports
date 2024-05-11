from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorWithPadding, DataCollatorForLanguageModeling, get_scheduler, HfArgumentParser
from datasets import Dataset, load_from_disk
import pandas as pd
import torch
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from tqdm.auto import tqdm

def main():
    retr_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-zh-v1.5')
    retr_model = AutoModel.from_pretrained('BAAI/bge-base-zh-v1.5')
    infer_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    infer_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
    )

    retr_model.train()
    infer_model.eval()

    def compute_embeddings(model, documents):
        # Tokenize sentences
        # encoded_input = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        model_output = model(**documents)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    INSTRUCTION = '''Given a list of functions with their documentation, call the correct function with the correct parameters in the form function_name(parameter 1, parameter 2).  Do not add any other text apart from the function call. If you cannot resolve the request with the given functions, call irrelevant_function() as a default.
    Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk').  Here is the documentation of all the functions.
    '''
    prompt_template = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION}{{}}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: {{}} Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{{}}{infer_tokenizer.eos_token}'

    dataset = load_from_disk('/proj/mounted/overlapping-functions-dataset')
    func_text = None
    with open('/proj/mounted/documentation.txt', "r") as file1:
        func_text = file1.read()

    documents = func_text.split("\n\n\n")
    tokenized_documents = retr_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

    # for inner data preparation
    inner_data_collator = DataCollatorForLanguageModeling(tokenizer=infer_tokenizer, mlm=False)

    def inner_tokenize_function(samples):
        return infer_tokenizer(samples["text"], truncation=True, padding=True, return_tensors="pt")

    def tokenize_function(samples):
        return retr_tokenizer(samples["query"], padding=True, truncation=True, return_tensors='pt')

    input_training_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    batch_size=8
    num_epochs=3
    k = 3
    gamma = 1
    beta = 1
    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    data_loader = DataLoader(
        input_training_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(retr_model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    """
    (B batch size, L length in tokens of each encoded example in a batch, V vocabulary length)
    loop for each batch:
    1. batch contains the tokenized queries by the encoder, use the batchs's indices in the loop to retrieve the queries
    2. re-compute the embeddings, retrieve top-k for each input query, for each query compute distribution Pr(d|x)
    3. for each query and each retrived document, make a prompt
    4. make k batches, each one containing the queries of the original batch with different documents
    5. run the model k times, once of each batch, obtain k outputs of size (B x Lk x V)
    6. compute perplexities for each input example of the k batches, get k vectors of length B
    7. stack the k vectors (on the colums) to get a matrix of shape (B x k), containing the perplexities for each input query on all of its retrieved documents
    8. use the perplexities on each input query to compute the distributions Q(d|x,y)
    9. compute the KL divergence between the distributions Pr(d|x) and Q(d|x,y) and average over all input queries
    """

    for epoch in range(num_epochs):
        for index, batch in tqdm(enumerate(data_loader)):
            # 1.
            batch_data = dataset["train"][index*batch_size:(index+1)*batch_size]
            # 2.
            embedded_documents = compute_embeddings(retr_model, tokenized_documents)
            embedded_queries = compute_embeddings(retr_model, batch)
            embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
            embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
            cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
            top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
            documents_per_query = top_k_docs.indices
            similarities_per_query = top_k_docs.values
            Pr = F.softmax(similarities_per_query / gamma, dim=1)
            # 3.
            prompts = [prompt_template.format(documents[doc_index], batch_data["query"][data_index], batch_data["response"][data_index])
            for i_th_doc in range(documents_per_query.size(1))
            for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])]
            # 4.
            inner_dataset = Dataset.from_pandas(pd.DataFrame(prompts, columns=["text"]))
            inner_dataset = inner_dataset.map(
                inner_tokenize_function,
                batched=True,
                remove_columns=inner_dataset.column_names
            )
            inner_data_loader = data_loader = DataLoader(
                inner_dataset, shuffle=False, batch_size=batch_size, collate_fn=inner_data_collator
            )
            # 5., 6. and 7.
            perplexities = []
            for inner_batch in inner_data_loader:
                labels = inner_batch.pop("labels")
                with torch.no_grad():
                    outputs = infer_model(**inner_batch)
                logits = outputs["logits"]
                # logits = torch.randn(batch_size, inner_batch["input_ids"].size(1), 256000)
                # labels = torch.randint(0, 256000, (batch_size, inner_batch["input_ids"].size(1)))
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                elem_wise_loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_per_sample = elem_wise_loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
                perplexity_per_sample = torch.exp(loss_per_sample)
                perplexities.append(perplexity_per_sample)
            perplexities = torch.stack(perplexities).T
            # 8.
            Q = F.softmax(perplexities / beta, dim=1)
            # Q = F.softmax(torch.randint(0, 3, (batch_size,k)).float(), dim=1)
            # 9.
            Q_log = torch.log(Q)
            divergence = kl_div(Q_log, Pr).sum(-1)
            loss = divergence.mean()
            print("BACKWARD STARTING...")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    torch.save(retr_model.state_dict(), '/proj/mounted/retr_model.pth')

if __name__ == "__main__":
    main()

