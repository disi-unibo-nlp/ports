import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset

"""Load the data and split into test and train (each function appears in both)"""

data = pd.read_csv("/proj/mounted/fine-tuning-no-irrelevant-func.csv", sep=";")

data_train = data[data.index % 10 < 8]

data_test = data[data.index % 10 >= 8]

"""Convert into DatasetDict"""

ds = DatasetDict()

dataset_train = Dataset.from_pandas(data_train, preserve_index=False)
dataset_test = Dataset.from_pandas(data_test, preserve_index=False)

ds['train'] = dataset_train
ds['test'] = dataset_test

"""Create Dataset with 17 functions' examples in the training set and 4 in the test set"""

ds2 = load_dataset("csv", data_files="/proj/mounted/fine-tuning-no-irrelevant-func.csv", sep=";")
ds2 = ds2["train"].train_test_split(test_size=0.19, shuffle=False)

"""Save both to disk"""
ds.save_to_disk("/proj/mounted/overlapping-functions-dataset-no-ir")
ds2.save_to_disk("/proj/mounted/non-overlapping-functions-dataset-no-ir")
