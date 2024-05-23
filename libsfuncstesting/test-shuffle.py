from datasets import load_from_disk

# Load the datasets
overlapping_dataset = load_from_disk("/proj/mounted/overlapping-functions-dataset")
non_overlapping_dataset = load_from_disk("/proj/mounted/non-overlapping-functions-dataset")

# Shuffle the datasets
overlapping_dataset_shuffled = overlapping_dataset.shuffle(seed=42)
non_overlapping_dataset_shuffled = non_overlapping_dataset.shuffle(seed=42)

print(overlapping_dataset["test"]["response"][:10])
print("-----")
print(overlapping_dataset_shuffled["test"]["response"][:10])

# Save the shuffled datasets
overlapping_dataset.save_to_disk("/proj/mounted/overlapping-functions-dataset-shuffled")
non_overlapping_dataset.save_to_disk("/proj/mounted/non-overlapping-functions-dataset-shuffled")

