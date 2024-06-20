import pandas as pd
import random
from datasets import Dataset, DatasetDict

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv('/proj/mounted/datasets/toole-single-tool.csv')

# Step 2: Identify unique tools in the 'response' column
unique_tools = df['response'].unique()

# Step 3: Randomly select 160 tools for one dataset and the remaining 40 for the other
random.shuffle(unique_tools)
tools_160 = unique_tools[:160]
tools_40 = unique_tools[160:]

# Step 4: Filter the DataFrame based on the selected tools
df_160 = df[df['response'].isin(tools_160)]
df_40 = df[df['response'].isin(tools_40)]

# Step 5: Convert the filtered DataFrames to Hugging Face Datasets
ds = DatasetDict()
dataset_train = Dataset.from_pandas(df_160, preserve_index=False)
dataset_test = Dataset.from_pandas(df_40, preserve_index=False)
ds['train'] = dataset_train
ds['test'] = dataset_test
ds.save_to_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")
