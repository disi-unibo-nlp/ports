import os
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding,
    # DataCollatorForLanguageModeling, 
    BitsAndBytesConfig,
    get_scheduler, 
    HfArgumentParser
)
from trl import DataCollatorForCompletionOnlyLM
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
from src.prompts import PROMPTS
import logging
import wandb

def main():
    # parse arguments and log to cloud services
    parser = HfArgumentParser(PyTorchTrainingParams)
    (args,) = parser.parse_args_into_dataclasses()
    log_wandb = args.log_to_wandb

    hf_key = os.getenv('HF_KEY')
    login(token=hf_key)

    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('/proj/mounted/log.out', mode='w')
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    def log_mem_usage(topic):
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device)
        memory_reserved = torch.cuda.memory_reserved(current_device)
        # logger.info(f"{topic} A: {memory_allocated / 1024**2} MB, R: {memory_reserved / 1024**2} MB")

    # load models and tokenizers
    retr_model_name = args.retr_model_name_or_path
    infer_model_name = args.infer_model_name_or_path
    retr_tokenizer = AutoTokenizer.from_pretrained(retr_model_name)
    retr_model = AutoModel.from_pretrained(retr_model_name).to("cuda")
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name, trust_remote_code=True)
    if infer_tokenizer.pad_token is None:
        logger.info("No padding token - using EOS instead")
        infer_tokenizer.pad_token = infer_tokenizer.eos_token
    infer_tokenizer.padding_side='left'
    if args.quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False if args.quantization_4bit else True,
            load_in_4bit=True if args.quantization_4bit else False,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16
        )
        infer_model = AutoModelForCausalLM.from_pretrained(
            infer_model_name,
            # torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        infer_model = AutoModelForCausalLM.from_pretrained(
            infer_model_name, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to("cuda")

    logger.info(f"Model dtype: {next(infer_model.parameters()).dtype}")

    # initialize wandb run
    if log_wandb:
        wandb_key = os.getenv('WANDB_KEY')
        name = args.wandb_proj_name
        wandb.login(key=wandb_key)
        wandb.init(
            project='fine-tuning-retriever',
            name=name if name else f"training run - {args.dataset_path.split('/')[-1]}",
            config={
                "retr_model_name_or_path": args.retr_model_name_or_path,
                "infer_model_name_or_path": args.infer_model_name_or_path,
                "quantize": args.quantize,
                "quantization_4bit": args.quantization_4bit,
                "batch_size": args.batch_size,
                "num_train_epochs": args.num_train_epochs,
                "num_retrieved_docs_per_query": args.num_retrieved_docs_per_query,
                "gamma_value": args.gamma_value,
                "beta_value": args.beta_value,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "dataset_path": args.dataset_path,
            }
        )
        wandb.watch(retr_model, log_freq=10)

    retr_model.eval()
    infer_model.eval()

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

    # data preparation
    INSTRUCTION = (
        "Given a list of functions with their documentation, call the correct function "
        "with the correct parameters in the form function_name(parameter 1, parameter 2). "
        "Do not add any other text apart from the function call.\n"
        "Example: Can you add a note saying 'Rembember the milk'? Response: add_note('Remember the milk'). "
        "Here is the documentation of all the functions."
    )
    # prompt_template = (
    #     f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION} {{}}'
    #     f'<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuery: {{}} Response:<|eot_id|>'
    #     f'<|start_header_id|>assistant<|end_header_id|>\n\n{{}}{infer_tokenizer.eos_token}'
    # )
    # ANSWER = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    infer_model_type = args.infer_model_type
    prompt_template = PROMPTS[infer_model_type]["prompt_template"]
    ANSWER = PROMPTS[infer_model_type]["answer_template"]
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.shuffle(seed=42).flatten_indices()
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

    input_eval_dataset = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    batch_size = args.batch_size
    num_epochs = args.num_train_epochs
    k = args.num_retrieved_docs_per_query
    gamma = args.gamma_value
    beta = args.beta_value
    data_collator = DataCollatorWithPadding(tokenizer=retr_tokenizer)
    train_data_loader = DataLoader(
        input_training_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )
    eval_data_loader = DataLoader(
        input_eval_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(retr_model.parameters(), lr=args.learning_rate)
    num_training_steps = num_epochs * len(train_data_loader)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    cross_entropy = CrossEntropyLoss(reduction='none')
    kl_div = KLDivLoss(reduction='none')

    # inner data utilities
    inner_data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ANSWER, 
        tokenizer=infer_tokenizer, 
        mlm=False)
    # inner_data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=infer_tokenizer, 
    #     mlm=False)

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
    
    # training utility functions
    def parse_batch(dataset_split, batch, index):
        """
        From the batch tokenized by the retriever's tokenizer, get the queries and responses in string format
        """
        curr_bs = batch["input_ids"].size(0)
        batch_data = dataset_split[index*batch_size:(index*batch_size)+curr_bs]
        return curr_bs, batch_data


    def get_top_k_docs_per_query(embedded_documents, batch, k):
        """
        Compute the top-k documents for each query in the batch, based on their cosine similarity
        """
        embedded_queries = compute_embeddings(retr_model, batch)
        embedded_documents_exp = embedded_documents.unsqueeze(0)  # Size becomes [1, n_docs, embeddings_size]
        embedded_queries_exp = embedded_queries.unsqueeze(1)  # Size becomes [batch_size, 1, embeddings_size]
        cos_sim = F.cosine_similarity(embedded_documents_exp, embedded_queries_exp, dim=-1)  # Size becomes [batch_size, n_docs]
        top_k_docs = torch.topk(cos_sim, k, dim=-1)  # Size becomes [batch_size, k]
        return top_k_docs.indices, top_k_docs.values

    compute_Pr = lambda similarities, gamma: F.softmax(similarities / gamma, dim=1)

    def get_prompts(prompt_template, documents, batch_data, documents_per_query):
        """
        Makes prompts to pass to the inference model using the respective documents for each query
        """
        prompts = [
                prompt_template.format(
                    INSTRUCTION,
                    documents[doc_index], 
                    batch_data[query_column][data_index], 
                    batch_data[response_column][data_index]
                )
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
            ]
        return prompts

    def prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator):
        """
        From the prompts, prepares the data to pass to the inference model
        Makes k batches of size curr_bs
        """
        inner_dataset = Dataset.from_pandas(pd.DataFrame(prompts, columns=["text"]))
        inner_dataset = inner_dataset.map(
            inner_tokenize_function,
            batched=True,
            remove_columns=inner_dataset.column_names
        )
        inner_data_loader = DataLoader(
            inner_dataset, shuffle=False, batch_size=curr_bs, collate_fn=inner_data_collator
        )
        return inner_data_loader

    def get_perplexity_per_sample(outputs, labels, cross_entropy, index, inner_index, input_ids):
        """
        From the inference model's outputs and the labels, compute the perplexity of each example
        """
        logits = outputs["logits"]
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        # log data of a specific example
        # if index == 1 and inner_index == 1:
        #     print("-----------------------")
        #     curr_logits = shift_logits[3]
        #     curr_labels = shift_labels[3]
        #     start_id = -8
        #     print(f"ORIGINAL TEXT: {infer_tokenizer.decode(input_ids[3], skip_special_tokens=False)}")
        #     print(f"LABEL TEXT: {infer_tokenizer.convert_ids_to_tokens(curr_labels[start_id:-1], skip_special_tokens=False)}")
        #     print(f"PRED TEXT: {infer_tokenizer.convert_ids_to_tokens(torch.argmax(curr_logits[start_id:-1], dim=1), skip_special_tokens=False)}")
        #     print(f"LABELS: {curr_labels}")
        #     print(f"PREDS: {torch.argmax(curr_logits, dim=1)}")
        #     curr_loss = cross_entropy(curr_logits, curr_labels)
        #     # print(f"LOSS: {curr_loss}")
        #     print(f"LOSS: {curr_loss[start_id:-1]}")
        #     num_elems = torch.sum(curr_labels.ne(-100))
        #     loss_mean = curr_loss.sum() / num_elems
        #     print(f"MEAN LOSS: {loss_mean}")
        #     print(f"PERPLEXITY: {torch.exp(loss_mean)}")
        #     print("***********************")

        elem_wise_loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_sum_per_sample = elem_wise_loss.view(shift_logits.size(0), shift_logits.size(1)).sum(dim=1)
        num_elems_per_sample = torch.sum(shift_labels.ne(-100), dim=1)
        loss_per_sample = loss_sum_per_sample / num_elems_per_sample 
        perplexity_per_sample = -torch.exp(loss_per_sample)
        print(f"PERPLEXITY PER SAMPLE: {-perplexity_per_sample}")
        return perplexity_per_sample

    def compute_Q(perplexities, beta):
        perplexities = torch.stack(perplexities).T
        Q = F.softmax(perplexities / beta, dim=1)
        # print(f"Q:\n{Q}")
        # exit()
        return Q

    def compute_loss(Q, Pr, kl_div):
        """
        Computes KL(Pr||Q)
        (Q and P are inverted in the function parameters, because it's how pytorch wants them)
        """
        Q_log = torch.log(Q)
        divergence = kl_div(Q_log, Pr).sum(-1)
        loss = divergence.mean()
        return loss

    # evaluation utility functions
    verify_relevancy = lambda response, doc: response.split('(')[0] in doc

    def get_relevant_docs(batch_data, documents):
        """
        For each example in the batch, retrieve its relevant document
        """        
        rel_docs = []
        bs = len(batch_data["response"])
        for response in batch_data["response"]:
            for i, doc in enumerate(documents):
                if verify_relevancy(response, doc):
                    rel_docs.append(i)
                    break
        return torch.tensor(rel_docs).unsqueeze(-1)

    def accumulate_ranks_at_n(ranks, documents_per_query, documents, batch_data, k, index):
        """
        Compute the ranks@n for the batch, sum them up to the previous batches' values
        """
        rel = get_relevant_docs(batch_data, documents).to("cuda")
        # if index == 0:
        logger.info(f"DOCS PER QUERY\n{documents_per_query}")
        logger.info(f"RELEVANT DOCS\n{rel}")
        for n in range(k):
            rank_at_n = torch.any(documents_per_query[:, :n+1] == rel, dim=-1).sum()
            ranks[n] += rank_at_n
        return ranks

    # evaluation function
    def evaluate(embedded_documents):
        with torch.no_grad():
            ranks = [0 for _ in range(k)]
            for index, batch in enumerate(eval_data_loader):
                # 1.
                curr_bs, batch_data = parse_batch(dataset["test"], batch, index)
                # 2.
                documents_per_query, similarities_per_query = get_top_k_docs_per_query(embedded_documents, batch, k)
                accumulate_ranks_at_n(ranks, documents_per_query, documents, batch_data, k, index)
                Pr = compute_Pr(similarities_per_query, gamma)
                del similarities_per_query
                # 3.
                prompts = get_prompts(prompt_template, documents, batch_data, documents_per_query)
                # 4.
                inner_data_loader = prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator)
                # 5., 6. and 7.
                perplexities = []
                for inner_index, inner_batch in enumerate(inner_data_loader):
                    inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                    labels = inner_batch.pop("labels")
                    with torch.no_grad():
                        outputs = infer_model(**inner_batch)
                    perplexity = get_perplexity_per_sample(outputs, labels, cross_entropy, index, inner_index, inner_batch["input_ids"])
                    perplexities.append(perplexity)
                    del outputs, perplexity
                    torch.cuda.empty_cache()
                # 8.
                Q = compute_Q(perplexities, beta)
                # 9.
                loss = compute_loss(Q, Pr, kl_div)
                assert not torch.isnan(loss), "Loss is NaN"
                assert loss < 1e6, f"Loss is too large: {loss}"
                del perplexities, Q, Pr, inner_data_loader, inner_batch
                torch.cuda.empty_cache()
                if log_wandb:
                    wandb.log({"Evaluation Loss": loss.item()})
                logger.info(f"EVALUATION LOSS: {loss}")
            ranks = [r / len(eval_data_loader.dataset) * 100 for r in ranks]
            for n in range(k):
                logger.info(f"RANK@{n+1}: {ranks[n]}")
                if log_wandb:
                    wandb.log({f"Rank@{n+1}": ranks[n] for n in range(k)})    

    with torch.no_grad():
        embedded_documents = compute_embeddings(retr_model, tokenized_documents)
    evaluate(embedded_documents)

    retr_model.train()
    for epoch in range(num_epochs):
        for index, batch in enumerate(train_data_loader):
            log_mem_usage("BATCH STARTING")
            # 1.
            curr_bs, batch_data = parse_batch(dataset["train"], batch, index)
            # 2.
            embedded_documents = compute_embeddings(retr_model, tokenized_documents)
            # documents_per_query contains the indices of the top-k documents for each query in the batch
            # similarities_per_query contains the cosine similarities of the top-k documents for each query in the batch
            documents_per_query, similarities_per_query = get_top_k_docs_per_query(embedded_documents, batch, k)
            Pr = compute_Pr(similarities_per_query, gamma)
            del similarities_per_query
            # 3.
            prompts = get_prompts(prompt_template, documents, batch_data, documents_per_query)
            # 4.
            inner_data_loader = prepare_inner_data_loader(prompts, curr_bs, inner_tokenize_function, inner_data_collator)
            # 5., 6. and 7.
            perplexities = []
            for inner_index, inner_batch in enumerate(inner_data_loader):
                inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                labels = inner_batch.pop("labels")
                with torch.no_grad():
                    outputs = infer_model(**inner_batch)
                perplexity = get_perplexity_per_sample(outputs, labels, cross_entropy, index, inner_index, inner_batch["input_ids"])
                perplexities.append(perplexity)
                del outputs, perplexity
                torch.cuda.empty_cache()
            # 8.
            Q = compute_Q(perplexities, beta)
            # 9.
            loss = compute_loss(Q, Pr, kl_div)
            log_mem_usage(f"BACKWARD STARTING WITH LOSS {loss}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            assert not torch.isnan(loss), "Loss is NaN"
            assert loss < 1e6, f"Loss is too large: {loss}"
            del perplexities, Q, Pr, inner_data_loader, inner_batch
            torch.cuda.empty_cache()
            if log_wandb:
                wandb.log({"Training Loss": loss.item()})
        retr_model.eval()
        evaluate(embedded_documents)
        retr_model.train()
        del embedded_documents
        torch.cuda.empty_cache()

    logger.info("TRAINING FINISHED.")
    if log_wandb:
        wandb.finish()
    # torch.save(retr_model.state_dict(), args.trained_model_save_path)

if __name__ == "__main__":
    main()

