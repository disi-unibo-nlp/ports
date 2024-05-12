import os
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding, 
    DataCollatorForLanguageModeling, 
    BitsAndBytesConfig,
    get_scheduler, 
    HfArgumentParser
)
from datasets import Dataset, load_from_disk
import pandas as pd
import torch
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss
from tqdm.auto import tqdm
from huggingface_hub import login
from src.data_classes import PyTorchTrainingParams
import logging

def main():
    parser = HfArgumentParser(PyTorchTrainingParams)
    (args,) = parser.parse_args_into_dataclasses()

    hf_key = os.getenv('HF_KEY')
    login(token=hf_key)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    def log_mem_usage():
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        logger.info(f"Memory Allocated: {memory_allocated / 1024**2} MB")
        logger.info(f"Memory Reserved: {memory_reserved / 1024**2} MB")

    retr_model_name = args.retr_model_name_or_path
    infer_model_name = args.infer_model_name_or_path
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model = AutoModel.from_pretrained(retr_model_name).to("cuda")
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    if infer_tokenizer.pad_token is None:
        print("No padding token - using EOS instead")
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    infer_model = AutoModelForCausalLM.from_pretrained(
        infer_model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )#.to("cuda")

    logger.info(f"Model dtype: {next(infer_model.parameters()).dtype}")
    log_mem_usage()

    retr_model.train()
    infer_model.eval()

    def compute_embeddings(model, documents):
        # Compute token embeddings
        documents = {k: v.to("cuda") for k, v in documents.items()}
        model_output = model(**documents)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    INSTRUCTION = (
        "Given a list of functions with their documentation, call the correct function "
        "with the correct parameters in the form function_name(parameter 1, parameter 2). "
        "Do not add any other text apart from the function call. If you cannot resolve the "
        "request with the given functions, call irrelevant_function() as a default.\n"
        "Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk'). "
        "Here is the documentation of all the functions."
    )

    prompt_template = (
        f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION}{{}}'
        f'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: {{}} Response:<|eot_id|>'
        f'<|start_header_id|>assistant<|end_header_id|>\n\n{{}}{infer_tokenizer.eos_token}'
    )

    dataset = load_from_disk(args.dataset_path)
    query_column = args.query_column
    response_column = args.response_column
    func_text = None
    with open(args.docs_path, "r") as f:
        func_text = f.read()

    documents = func_text.split("\n\n\n")
    tokenized_documents = retr_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

    def tokenize_function(samples):
        return retr_tokenizer(samples[query_column], padding=True, truncation=True, return_tensors='pt')

    input_training_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    batch_size = args.batch_size
    num_epochs = args.num_train_epochs
    k = args.num_retrieved_docs_per_query
    gamma = args.gamma_value
    beta = args.beta_value
    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    data_loader = DataLoader(
        input_training_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(retr_model.parameters(), lr=args.learning_rate)
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    # for inner data preparation
    inner_data_collator = DataCollatorForLanguageModeling(tokenizer=infer_tokenizer, mlm=False)

    def inner_tokenize_function(samples):
        return infer_tokenizer(samples["text"], truncation=True, padding=True, return_tensors="pt")

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
        for index, batch in enumerate(data_loader):
            logger.info("NEW BATCH STARTING...")
            # 1.
            batch_data = dataset["train"][index*batch_size:(index+1)*batch_size]
            # 2.
            embedded_documents = compute_embeddings(retr_model, tokenized_documents)
            if "labels" in batch:
                logger.warning(f"LABELS IN DOCUMENTS IN BATCH {index} EPOCH {epoch}")
            embedded_queries = compute_embeddings(retr_model, batch)
            embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
            embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
            cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
            top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
            documents_per_query = top_k_docs.indices
            similarities_per_query = top_k_docs.values
            Pr = F.softmax(similarities_per_query / gamma, dim=1)
            log_mem_usage()
            # 3.
            prompts = [
                prompt_template.format(
                    documents[doc_index], 
                    batch_data[query_column][data_index], 
                    batch_data[response_column][data_index]
                )
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
            ]
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
                inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                labels = inner_batch.pop("labels")
                with torch.no_grad():
                    outputs = infer_model(**inner_batch)
                log_mem_usage()
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
            logger.info("BACKWARD STARTING...")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    logger.info("TRAINING FINISHED.")
    # torch.save(retr_model.state_dict(), args.trained_model_save_path)

if __name__ == "__main__":
    main()

