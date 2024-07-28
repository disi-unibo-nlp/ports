import pandas as pd
from datasets import load_from_disk


octopus_over = load_from_disk("/proj/mounted/datasets/octopus-overlapping")
octopus_non_over = load_from_disk("/proj/mounted/datasets/octopus-non-overlapping")
toole_over = load_from_disk("/proj/mounted/datasets/toole-single-tool-overlapping")
toole_non_over = load_from_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")

# Verify dataset informations and consistency
print("Octopus Overlapping")
print("-" * 20)
octopus_over_train_size = len(octopus_over['train']['response'])
octopus_over_test_size = len(octopus_over['test']['response'])
octopus_over_total_size = octopus_over_train_size + octopus_over_test_size
print(f"Train size: {octopus_over_train_size}, Test size: {octopus_over_test_size}")
print(f"Train size percentage: {octopus_over_train_size / octopus_over_total_size * 100:.2f}%, Test size percentage: {octopus_over_test_size / octopus_over_total_size * 100:.2f}%")
octopus_over_tools_train = (pd.Series([el.split('(')[0] for el in octopus_over["train"]["response"]])).unique()
octopus_over_tools_test = (pd.Series([el.split('(')[0] for el in octopus_over["test"]["response"]])).unique()
print(f"Num tools in train: {len(octopus_over_tools_train)}, Num tools in test: {len(octopus_over_tools_test)}")
print(f"Common tools: {len(set(octopus_over_tools_train).intersection(set(octopus_over_tools_test)))}")
print()

print("Octopus Non-Overlapping")
print("-" * 20)
octopus_non_over_train_size = len(octopus_non_over['train']['response'])
octopus_non_over_test_size = len(octopus_non_over['test']['response'])
octopus_non_over_total_size = octopus_non_over_train_size + octopus_non_over_test_size
print(f"Train size: {octopus_non_over_train_size}, Test size: {octopus_non_over_test_size}")
print(f"Train size percentage: {octopus_non_over_train_size / octopus_non_over_total_size * 100:.2f}%, Test size percentage: {octopus_non_over_test_size / octopus_non_over_total_size * 100:.2f}%")
octopus_non_over_tools_train = (pd.Series([el.split('(')[0] for el in octopus_non_over["train"]["response"]])).unique()
octopus_non_over_tools_test = (pd.Series([el.split('(')[0] for el in octopus_non_over["test"]["response"]])).unique()
print(f"Num tools in train: {len(octopus_non_over_tools_train)}, Num tools in test: {len(octopus_non_over_tools_test)}")
print(f"Common tools: {len(set(octopus_non_over_tools_train).intersection(set(octopus_non_over_tools_test)))}")
print()

print("Toole Overlapping")
print("-" * 20)
toole_over_train_size = len(toole_over['train']['response'])
toole_over_test_size = len(toole_over['test']['response'])
toole_over_total_size = toole_over_train_size + toole_over_test_size
print(f"Train size: {toole_over_train_size}, Test size: {toole_over_test_size}")
print(f"Train size percentage: {toole_over_train_size / toole_over_total_size * 100:.2f}%, Test size percentage: {toole_over_test_size / toole_over_total_size * 100:.2f}%")
toole_over_tools_train = (pd.Series(toole_over["train"]["response"])).unique()
toole_over_tools_test = (pd.Series(toole_over["test"]["response"])).unique()
print(f"Num tools in train: {len(toole_over_tools_train)}, Num tools in test: {len(toole_over_tools_test)}")
print(f"Common tools: {len(set(toole_over_tools_train).intersection(set(toole_over_tools_test)))}")
print()

print("Toole Non-Overlapping")
print("-" * 20)
toole_non_over_train_size = len(toole_non_over['train']['response'])
toole_non_over_test_size = len(toole_non_over['test']['response'])
toole_non_over_total_size = toole_non_over_train_size + toole_non_over_test_size
print(f"Train size: {toole_non_over_train_size}, Test size: {toole_non_over_test_size}")
print(f"Train size percentage: {toole_non_over_train_size / toole_non_over_total_size * 100:.2f}%, Test size percentage: {toole_non_over_test_size / toole_non_over_total_size * 100:.2f}%")
toole_non_over_tools_train = (pd.Series(toole_non_over["train"]["response"])).unique()
toole_non_over_tools_test = (pd.Series(toole_non_over["test"]["response"])).unique()
print(f"Num tools in train: {len(toole_non_over_tools_train)}, Num tools in test: {len(toole_non_over_tools_test)}")
print(f"Common tools: {len(set(toole_non_over_tools_train).intersection(set(toole_non_over_tools_test)))}")

