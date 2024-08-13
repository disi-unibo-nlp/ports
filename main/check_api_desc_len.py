from datasets import load_dataset
from huggingface_hub import login
import os
login(os.getenv('HF_KEY'))
datasets = [
    'ToolRetriever/ToolBench',
    'ToolRetriever/APIBench',
    'ToolRetriever/APIBank',
    'ToolRetriever/OctopusOverlapping',
    'ToolRetriever/ToolEOverlapping',
]

for dtst in datasets:
    dataset = load_dataset(dtst, 'parsed_data')
    m_len = max(set([len(x) for x in dataset['train']['api_description']]))
    print(f'{dtst}: {m_len}')

dataset = load_dataset('ToolRetriever/BFCL', 'parsed_data')
m_len = max(set([len(x) for x in dataset['test']['api_description']]))
print(f'BFCL: {m_len}')
