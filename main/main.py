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
import wandb

def main():
    CORRECT_LOGIT_SCORE = 1e10
    WRONG_LOGIT_SCORE = -1e10
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
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    terminators = [
        infer_tokenizer.eos_token_id,
        infer_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
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
            quantization_config=quantization_config
        )
    else:
        infer_model = AutoModelForCausalLM.from_pretrained(infer_model_name, torch_dtype=torch.bfloat16).to("cuda")

    logger.info(f"Model dtype: {next(infer_model.parameters()).dtype}")

    # initialize wandb run
    if log_wandb:
        wandb_key = os.getenv('WANDB_KEY')
        wandb.login(key=wandb_key)
        wandb.init(
            project='fine-tuning-retriever',
            name=f"training run - {args.dataset_path.split('/')[-1]}",
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

    prompt_template = (
        f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{INSTRUCTION} {{}}<|eot_id|>'
        f'<|start_header_id|>user<|end_header_id|>\n\nQuery: {{}} Response:<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )

    response_template = '{}<|eot_id|>'

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
    gen_config = {
        'do_sample':   False,
        'max_new_tokens' : 0, # set each call
        'num_beams' : 1,
        'use_cache' : True,
        'pad_token_id' : infer_tokenizer.eos_token_id,
        # 'temperature': 0,
        # 'top_p': 0,
        'output_logits': True,
        'return_dict_in_generate': True,
        'eos_token_id': terminators,
    }
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
    def inner_tokenize_function(samples):
        res = infer_tokenizer(samples["prompt_text"], add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")
        infer_tokenizer.padding_side='right'
        labels = infer_tokenizer(samples["labels_text"], add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")
        res["labels"] = labels["input_ids"]
        # res["labels_attention_mask"] = labels["attention_mask"]
        infer_tokenizer.padding_side='left'
        return res

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

    def get_gen_data(documents, batch_data, documents_per_query):
        """
        Return the textual data for generation with the inference model, i.e prompts and responses
        to pass to the inference model using the respective documents for each query
        """
        prompts = [
                prompt_template.format(
                    documents[doc_index], 
                    batch_data[query_column][data_index], 
                )
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
            ]
        responses = [
                response_template.format(batch_data[response_column][data_index])
                for i_th_doc in range(documents_per_query.size(1))
                for data_index, doc_index in enumerate(documents_per_query[:, i_th_doc])
        ]
        return prompts, responses

    def prepare_inner_data_loader(prompts, responses, curr_bs, inner_tokenize_function, data_collator):
        """
        From the prompts, prepares the data to pass to the inference model
        Makes k batches of size curr_bs
        """
        inner_dataset = Dataset.from_pandas(pd.DataFrame({"prompt_text": prompts, "labels_text": responses}))
        inner_dataset = inner_dataset.map(
            inner_tokenize_function,
            batched=True,
            remove_columns=inner_dataset.column_names
        )
        inner_data_loader = DataLoader(
            inner_dataset, shuffle=False, batch_size=curr_bs, collate_fn=data_collator
        )
        return inner_data_loader

    def parse_outputs(outputs, labels):
        """
        Adds padding tensors to logits to match the target length, corrects the logits after the first eot_id
        """
        target = labels.size(1)
        # Logits are a list of num_tokens tensors, each tensor has size [batch_size, vocabulary_size]
        logits = list(outputs["logits"])

        b, v = logits[0].size()
        num_tokens = len(logits)
        # Set to eot_id the tokens generated after the first eot_id
        # eot_vector = torch.full((v,), WRONG_LOGIT_SCORE).to("cuda")
        # eot_vector[infer_tokenizer.eos_token_id] = CORRECT_LOGIT_SCORE
        eot_vector = None
        eot_reached = False
        for example in range(b):
            for token in range(num_tokens):
                if eot_reached:
                    logits[token][example] = eot_vector.clone()
                elif torch.argmax(logits[token][example]) == infer_tokenizer.eos_token_id:
                    eot_reached = True
                    eot_vector = logits[token][example].clone()
            eot_reached = False
        # Add padding to the logits to match the target length
        padding_tensor = logits[-1]
        logits.extend([padding_tensor.clone()] * (target - len(logits)))
        # for _ in range(target - len(logits)):
        #     # # Create a new tensor of size [b, v] with all zeros
        #     # new_tensor = torch.full((b, v), WRONG_LOGIT_SCORE).to("cuda")
        #     # # Set the value at the eos_token_id position to 1
        #     # new_tensor.scatter_(1, torch.full((b, 1), infer_tokenizer.eos_token_id).to("cuda"), CORRECT_LOGIT_SCORE)
        #     logits.append(padding_tensor.clone())
        full_logits = torch.stack(logits).transpose(0, 1)
        # print(f"LOGITS LABELS MATCH SIZE: {full_logits.size(1) == labels.size(1)}")
        # print(f"LOGITS SIZE: {full_logits.size()}, LABELS SIZE: {labels.size()}")
        # for i in range(len(outputs["sequences"])):
        #     print(f"SEQ {i}: {outputs['sequences'][i][-2]}")
        return full_logits


    def get_perplexity_per_sample(logits, labels, cross_entropy):
        """
        From the inference model's outputs and the labels, compute the perplexity of each example
        """
        labels = labels.contiguous()
        logits = logits.contiguous()

        print("-----------------------")
        curr_logits = logits[1]
        curr_labels = labels[1]
        print(f"LABELS: {curr_labels}")
        print(f"PREDS: {torch.argmax(curr_logits, dim=1)}")
        print(f"LABELS TEXT: {infer_tokenizer.decode(curr_labels, skip_special_tokens=False)}")
        print(f"PREDS TEXT: {infer_tokenizer.decode(torch.argmax(curr_logits, dim=1), skip_special_tokens=False)}")
        curr_loss = cross_entropy(curr_logits, curr_labels)
        print(f"LOSS: {curr_loss}")
        loss_mean = curr_loss.mean()
        print(f"MEAN LOSS: {loss_mean}")
        print(f"PERPLEXITY: {torch.exp(loss_mean)}")
        print("***********************")
        # exit()

        elem_wise_loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss_per_sample = elem_wise_loss.view(logits.size(0), logits.size(1)).mean(axis=1)
        perplexity_per_sample = torch.exp(loss_per_sample)
        logger.info(f"PERPLEXITY: {perplexity_per_sample}")
        # exit()
        return perplexity_per_sample

    def compute_Q(perplexities, beta):
        perplexities = torch.stack(perplexities).T
        # logger.info(f"PERPLEXITIES SIZE: {perplexities.size()}")
        # logger.info(f"PERPLEXITIES MATRIX:\n{perplexities}")
        Q = F.softmax(perplexities / beta, dim=1)
        logger.info(f"Q:\n{Q}")
        return Q

    def compute_loss(Q, Pr, kl_div):
        """
        Computes KL(Pr||Q)
        (Q and P are inverted in the function parameters, because it's how pytorch wants them)
        """
        # logger.info(f"Q:\n{Q}")
        # logger.info(f"Pr:\n{Pr}")
        Q_log = torch.log(Q)
        divergence = kl_div(Q_log, Pr).sum(-1)
        logger.info(f"DIVERGENCE:\n{divergence}")
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
        return torch.tensor(rel_docs).unsqueeze(-1)

    def accumulate_ranks_at_n(ranks, documents_per_query, documents, batch_data, k, index):
        """
        Compute the ranks@n for the batch, sum them up to the previous batches' values
        """
        rel = get_relevant_docs(batch_data, documents).to("cuda")
        if index == 0:
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
                prompts, responses = get_gen_data(documents, batch_data, documents_per_query)
                # 4.
                inner_data_loader = prepare_inner_data_loader(prompts, responses, curr_bs, inner_tokenize_function, data_collator)
                # 5., 6. and 7.
                perplexities = []
                for inner_batch in inner_data_loader:
                    inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                    labels = inner_batch.pop("labels")
                    # labels_attention_mask = inner_batch.pop("labels_attention_mask")
                    gen_config["max_new_tokens"] = labels.size(1)
                    with torch.no_grad():
                        outputs = infer_model.generate(**inner_batch, **gen_config)
                    logits = parse_outputs(outputs, labels)
                    perplexity = get_perplexity_per_sample(logits, labels, cross_entropy)
                    perplexities.append(perplexity)
                    del outputs, perplexity
                    torch.cuda.empty_cache()
                exit()
                # 8.
                Q = compute_Q(perplexities, beta)
                # 9.
                loss = compute_loss(Q, Pr, kl_div)
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
            prompts, responses = get_gen_data(documents, batch_data, documents_per_query)
            # 4.
            inner_data_loader = prepare_inner_data_loader(prompts, responses, curr_bs, inner_tokenize_function, data_collator)
            # 5., 6. and 7.
            perplexities = []
            for inner_batch in inner_data_loader:
                inner_batch = {k: v.to("cuda") for k, v in inner_batch.items()}
                labels = inner_batch.pop("labels")
                gen_config["max_new_tokens"] = labels.size(1)
                with torch.no_grad():
                    outputs = infer_model.generate(**inner_batch, **gen_config)
                logits = parse_outputs(outputs, labels)
                perplexity = get_perplexity_per_sample(logits, labels, cross_entropy)
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

