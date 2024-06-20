import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/proj/mounted/datasets/toole-single-tool.csv')

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into training and testing sets
train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

dataset_dict.save_to_disk("/proj/mounted/datasets/toole-single-tool-dataset")
