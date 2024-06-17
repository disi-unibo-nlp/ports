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
    parser = HfArgumentParser(PyTorchTrainingParams)
    (args,) = parser.parse_args_into_dataclasses()
    hf_key = os.getenv('HF_KEY')
    login(token=hf_key)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('/proj/mounted/log.out', mode='w')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    infer_model_name = args.infer_model_name_or_path
    infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_name)
    infer_tokenizer.padding_side='left'
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False if args.quantization_4bit else True,
        load_in_4bit=True if args.quantization_4bit else False,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    )
    infer_model = AutoModelForCausalLM.from_pretrained(
        infer_model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    terminators = [
        infer_tokenizer.eos_token_id,
        infer_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    prompt_template = (
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a language model<|eot_id|>'
        '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    )
    queries = [
        prompt_template.format("Who are you?"),
        prompt_template.format("Hi, how are you? Answer briefly in just one word."),
    ]
    infer_tokenizer.pad_token = infer_tokenizer.eos_token
    tokenized_queries = infer_tokenizer(queries, add_special_tokens=False, padding=True, return_tensors="pt")
    gen_config = {
        'do_sample':   False,
        'max_new_tokens' : 100,
        'num_beams' : 1,
        'use_cache' : True,
        'pad_token_id' : infer_tokenizer.eos_token_id,
        # 'temperature': 0,
        # 'top_p': 0,
        'output_logits': True,
        'eos_token_id': terminators,
        'return_dict_in_generate': True,
    }
    gen_data = {'input_ids' : tokenized_queries['input_ids'].to("cuda"), 'attention_mask' : tokenized_queries['attention_mask'].to("cuda")}
    predictions = infer_model.generate(**gen_data, **gen_config)
    logger.info(f"keys: {predictions.keys()}")
    # logger.info(f"Queries: {queries}")
    # logger.info(f"Tokenized queries: {tokenized_queries}")
    # decoded = infer_tokenizer.batch_decode(predictions["sequences"][0], skip_special_tokens=False)
    # logger.info(f"Predictions: {decoded}")
    # logger.info(f"Tokenized predictions: {predictions['sequences']}")
    # print(infer_tokenizer.decode(predictions["sequences"][0], skip_special_tokens=False))







    # predicted_indices = torch.argmax(logits, dim=1)
    # print(predicted_indices)
    # decoded_from_logits = infer_tokenizer.convert_ids_to_tokens(predicted_indices)
    # print(decoded_from_logits)
    # print(predictions["sequences"].size())
    # decoded = infer_tokenizer.decode(predictions["sequences"][0], skip_special_tokens=False)
    # print(decoded)
    # decoded_t = infer_tokenizer.convert_ids_to_tokens(predictions["sequences"][0])
    # print(decoded_t)
    # print(infer_tokenizer.decode(predictions["sequences"][0], skip_special_tokens=False))



if __name__ == "__main__":
    main()

