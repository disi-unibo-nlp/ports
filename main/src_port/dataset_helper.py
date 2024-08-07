from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import List


class DatasetDownloader():

    def __init__(self,
                 dataset_name):
        super(self).__init__()

        self.dataset_name = dataset_name
        self.data_sub_split = "parsed_data"
        
        base_ds_path = "ToolRetriever"
        dataset_mapping = {
            "bfcl" : "BFCL",
            "apibank" : "APIBank",
            "apibench" : "APIBench",
            "octopus" : "OctopusNonOverlapping",
            "toole" : "ToolENonOverlapping",
            "toolbench" : "ToolBench"
        }

        data_path = "/".join([base_ds_path, dataset_mapping[dataset_name]])
        self.data_path = data_path

    def get_dataset(self):
        """
        Download and return the dataset
        """
        return load_dataset(self.data_path, self.data_sub_split)
    
    def post_process_answers(self,
                             dataset):
        """
        Parse the dataset's labels by adding answer parameters if needed according to they dataset type.
        """
        def __parse_apibank(example):
            out = example
            out["answer"] = f'{out["answer"]}({out["answer_params"]})'
            return out

        if self.dataset_name == "apibank":
            for k in dataset.keys():
                dataset[k] = dataset[k].map(__parse_apibank, remove_columns=dataset[k].column_names)

        return dataset


# --------------------------------------------------------
# >> DATA HELPERs

def get_prompted_q_doc(prompt_template : str,
                       document : str,
                       query : str,
                       answer : str,
                       instruction_prompt : str):
    """
    Define prompts to pass to the inference model
    """

    prompt_fields = {
        "instruction" : instruction_prompt,
        "retrieved_doc" : document,
        "query" : query,
        "answer" : answer
    }

    prompts = prompt_template.format(**prompt_fields)

    return prompts


def create_triplets_with_unique_multiple_negatives(dataset : Dataset,
                                                   num_negatives_per_positive : int = 1, 
                                                   split : str = 'train',
                                                   start_seed : int = 0):
    """
    Create triplets of (pseudo-) randomly sampled negatives
    """

    triplets = []
    questions = dataset[split]["query"]
    augmented_descriptions = dataset[split]["api_description"]
    answer = dataset[split]["answer"]


    for i in range(len(questions)):
        query = questions[i]
        positive = augmented_descriptions[i]

        # Create a list of possible negatives with their indices
        possible_negatives = list(enumerate(augmented_descriptions))
        possible_negatives = [item for item in possible_negatives if item[0] != i] # exclude self from negs

        seed_i = start_seed + i 
        random.seed(seed_i) # set incremental seed for reproducibility

        # Choose multiple unique negatives
        num_unique_negatives = min(num_negatives_per_positive, len(possible_negatives))
        negative_samples = random.sample(possible_negatives, num_unique_negatives)

        triplets.append({
            'query': query,
            'positive': positive,
            'negative': [n[1] for n in negative_samples],
            'pos_answer': answers[i],
            'neg_answer': [answers[n[0]] for n in negative_samples]
        })

    return triplets

def create_instances_wo_negs(dataset : Dataset,
                             split : str = 'train'):
    data = []
    questions = dataset[split]["query"]
    augmented_descriptions = dataset[split]["api_description"]
    answer = dataset[split]["answer"]

    for i, q in enumerate(questions):
        row = {
            "query" : q,
            "positive" : augmented_descriptions[i],
            "pos_answer" : answer[i]
        }
        data.append(row)

    return data


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

class TripletCollator(DataCollatorMixin):
    def __init__(self,
                 retrieval_tokenizer,
                 inference_tokenizer,
                 def_corpus=None,
                 prompt_template : str = "",
                 instruction_prompt = str = "",
                 max_length_retrieval=128,
                 max_length_generator=1024):
        self.retr_tokenizer = retrieval_tokenizer
        self.gen_tokenizer = inference_tokenizer
        self.corpus = def_corpus
        self.prompt_template = prompt_template
        self.instruction_prompt = instruction_prompt
        
        self.max_length_retr = max_length_retrieval
        self.max_length_gen = max_length_generator

    def __call__(self, batch):
        prompt_template = self.prompt_template
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item['negative'] for item in batch]

        pos_answers = [item['pos_answer'] for item in batch]
        neg_answers = [item['neg_answer'] for item in batch]

        # Queries, Pos, Neg docs
        query_encodings = self.retr_tokenizer(queries, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        positive_encodings = self.retr_tokenizer(positives, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        negative_encodings = [self.retr_tokenizer(neg, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt') for neg in negatives]

        # Prompts for inference
        prompted_q_pos = [get_prompted_q_doc(prompt_template=prompt_template, document=item['positive'], query=item['query'], answer=item['pos_answer'], instruction_prompt=self.instruction_prompt) for item in batch]
        prompted_q_pos_encodings = self.gen_tokenizer(prompted_q_pos,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt')

        prompted_q_neg = []
        for item in batch:
          sub_prompted_neg = []
          #print(item)
          #print(len(negatives[0]))
          for neg_i in range(len(negatives[0])):
            #print(item['negative'][neg_i])
            _p_neg = get_prompted_q_doc(prompt_template=prompt_template, document=item['negative'][neg_i], query=item['query'], answer=item['neg_answer'][neg_i], instruction_prompt=self.instruction_prompt)
            sub_prompted_neg.append(_p_neg)
          prompted_q_neg.append(sub_prompted_neg)


        prompted_q_neg_encodings = [self.gen_tokenizer(neg_subset,
                                                        truncation=True,
                                                        max_length=self.max_length_gen,
                                                        padding='max_length',
                                                        return_tensors='pt') for neg_subset in prompted_q_neg]

        # Labels
        labels_encodings_pos = self.gen_tokenizer(pos_answers,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt')
        labels_encodings_neg = [self.gen_tokenizer(neg_a,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt') for neg_a in neg_answers]

        # Remove token_type_ids
        for encoding in [query_encodings,
                         positive_encodings,
                         prompted_q_pos_encodings,
                         labels_encodings_pos]:
          encoding.pop('token_type_ids', None)

        for encoding in [negative_encodings,
                         prompted_q_neg_encodings,
                         labels_encodings_neg]:
          for el in encoding:
            el.pop('token_type_ids', None)

        # Get gold retrieval_ids wrt corpus
        gold_indices = []
        for pos_doc in positives:
          pos_idx = self.corpus.index(pos_doc)
          gold_indices.append(pos_idx)
        gold_indices = torch.tensor(gold_indices)


        return {
            'query': query_encodings,
            'positive': positive_encodings,
            'negative': negative_encodings,
            'q_pos_prompt' : prompted_q_pos_encodings,
            'q_neg_prompt' : prompted_q_neg_encodings,
            'labels_pos' : labels_encodings_pos,
            'labels_neg' : labels_encodings_neg,
            'gold_retrieval_ids' : gold_indices
        }


class EvalTripletCollator(DataCollatorMixin):
    def __init__(self,
                 retrieval_tokenizer : AutoTokenizer,
                 def_corpus : List[str] = None,
                 max_length_retrieval : int = 128):
        self.retr_tokenizer = retrieval_tokenizer
        self.corpus = def_corpus
        self.max_length_retr = max_length_retrieval

    def __call__(self, batch):
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]
        pos_answers = [item['pos_answer'] for item in batch]

        # Queries, Pos, Neg docs
        query_encodings = self.retr_tokenizer(queries, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')
        positive_encodings = self.retr_tokenizer(positives, truncation=True, max_length=self.max_length_retr, padding='max_length', return_tensors='pt')

        # Labels
        labels_encodings_pos = self.gen_tokenizer(pos_answers,
                                                  truncation=True,
                                                  max_length=self.max_length_gen,
                                                  padding='max_length',
                                                  return_tensors='pt')

        # Remove token_type_ids
        for encoding in [query_encodings,
                         positive_encodings,
                         labels_encodings_pos]:
          encoding.pop('token_type_ids', None)

        # Get gold retrieval_ids wrt corpus
        gold_indices = []
        for pos_doc in positives:
          pos_idx = self.corpus.index(pos_doc)
          gold_indices.append(pos_idx)
        gold_indices = torch.tensor(gold_indices)


        return {
            'query': query_encodings,
            'positive': positive_encodings,
            'labels_pos' : labels_encodings_pos,
            'gold_retrieval_ids' : gold_indices
        }



def get_train_dataloader(dataset,
                         api_corpus_list : List[str],
                         retrieval_tokenizer : AutoTokenizerm,
                         inference_tokenizer : AutoTokenizerm,
                         prompt_template : str = "",
                         retrieval_max_length : int = 514,
                         generateor_max_length : int = 1024,
                         batch_size : int = 2,
                         epoch_number : int = 0,
                         num_neg_examples : int = 2) -> torch.utils.data.DataLoader:
    """
    Create triplets and return train dataloader
    """
    triplets = create_triplets_with_unique_multiple_negatives(dataset,
                                                              num_negatives_per_positive=num_neg_examples,
                                                              split='train', 
                                                              start_seed = epoch_number)

    triplet_dataset = TripletDataset(triplets)

    train_collator = TripletCollator(retrieval_tokenizer=retrieval_tokenizer,
                                    inference_tokenizer=inference_tokenizer,
                                    prompt_template=prompt_template,
                                    def_corpus=api_corpus_list,
                                    max_length_retrieval=retrieval_max_length,
                                    max_length_generator=generateor_max_length)

    triplet_dataloader = DataLoader(triplet_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=train_collator)
                                    
    return triplet_dataloader


def get_eval_dataloader(dataset,
                        api_corpus_list : List[str],
                        retrieval_max_length : int = 514,
                        retrieval_tokenizer : AutoTokenizerm,
                        batch_size : int = 2) -> torch.utils.data.DataLoader:

    eval_data = create_instances_wo_negs(datset, split="test")
    eval_triplet_dataset = TripletDataset(eval_data)
    
    eval_collator = EvalTripletCollator(retrieval_tokenizer=retr_tokenizer,
                                        def_corpus=api_corpus_list,
                                        max_length=retrieval_max_length)

    eval_dataloader = DataLoader(eval_triplet_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=eval_collator)
    
    
    return eval_dataloader