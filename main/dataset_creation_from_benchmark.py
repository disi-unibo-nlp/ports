import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

"""Load the data and split into test and train (each function appears in both)"""

# data = pd.read_csv("/proj/mounted/datasets/fine-tuning-no-irrelevant-func.csv", sep=";")

# data_train = data[data.index % 10 < 8]

# data_test = data[data.index % 10 >= 8]

# """Convert into DatasetDict"""

# ds = DatasetDict()

# dataset_train = Dataset.from_pandas(data_train, preserve_index=False)
# dataset_test = Dataset.from_pandas(data_test, preserve_index=False)

# ds['train'] = dataset_train
# ds['test'] = dataset_test

# """Create Dataset with 16 functions' examples in the training set and 4 in the test set"""

# ds2 = load_dataset("csv", data_files="/proj/mounted/datasets/fine-tuning-no-irrelevant-func.csv", sep=";")
# ds2 = ds2["train"].train_test_split(test_size=0.2, shuffle=False)

# print((pd.Series([el.split('(')[0] for el in ds["test"]["response"]])).unique())

"""Save both to disk"""
# ds.save_to_disk("/proj/mounted/overlapping-functions-dataset-no-ir")
# ds2.save_to_disk("/proj/mounted/non-overlapping-functions-dataset-no-ir")

octopus_over = load_from_disk("/proj/mounted/datasets/overlapping-functions-dataset-no-ir")
octopus_non_over = load_from_disk("/proj/mounted/datasets/non-overlapping-functions-dataset-no-ir")
toole_over = load_from_disk("/proj/mounted/datasets/toole-single-tool-dataset")
toole_non_over = load_from_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")

print("Verify non-overlapping datasets")
print(((pd.Series([el.split('(')[0] for el in octopus_over["test"]["response"]])).unique() == (pd.Series([el.split('(')[0] for el in octopus_over["test"]["response"]])).unique()).all())
print(((pd.Series([el.split('(')[0] for el in toole_over["test"]["response"]])).unique() == (pd.Series([el.split('(')[0] for el in toole_over["test"]["response"]])).unique()).all())
print("verify toole")
print(toole_over)
print(toole_non_over)
toole_tools_train = (pd.Series(toole_non_over["train"]["response"])).unique()
toole_tools_test = (pd.Series(toole_non_over["test"]["response"])).unique()
print(any(el in toole_tools_train for el in toole_tools_test))
print(len(toole_tools_train), len(toole_tools_test))
print((pd.Series([el.split('(')[0] for el in toole_non_over["test"]["response"]])).unique())
print("verify octopus")
print(octopus_over)
print(octopus_non_over)
octopus_tools_train = (pd.Series([el.split('(')[0] for el in octopus_non_over["train"]["response"]])).unique()
octopus_tools_test = (pd.Series([el.split('(')[0] for el in octopus_non_over["test"]["response"]])).unique()
print(any(el in octopus_tools_train for el in octopus_tools_test))
print(len(octopus_tools_train), len(octopus_tools_test))
print((pd.Series([el.split('(')[0] for el in octopus_non_over["test"]["response"]])).unique())
