from datasets import load_from_disk, DatasetDict
from huggingface_hub import login 
import os
login(token=os.getenv("HF_KEY"))

octopus_dataset = load_from_disk("/proj/mounted/to_the_hub_datasets/OctopusNonOverlapping")
toole_dataset = load_from_disk("/proj/mounted/to_the_hub_datasets/ToolENonOverlapping")

octopus_train_test_dataset = DatasetDict({
    'train': octopus_dataset['train'],
    'test': octopus_dataset['test']
})
octopus_apis_dataset = DatasetDict({
    'APIs': octopus_dataset['APIs']
})
octopus_train_test_dataset.push_to_hub('ToolRetriever/OctopusNonOverlapping', config_name='train_test_data')
octopus_apis_dataset.push_to_hub('ToolRetriever/OctopusNonOverlapping', config_name='APIs')

toole_train_test_dataset = DatasetDict({
    'train': toole_dataset['train'],
    'test': toole_dataset['test']
})
toole_apis_dataset = DatasetDict({
    'APIs': toole_dataset['APIs']
})
toole_train_test_dataset.push_to_hub('ToolRetriever/ToolENonOverlapping', config_name='train_test_data')
toole_apis_dataset.push_to_hub('ToolRetriever/ToolENonOverlapping', config_name='APIs')