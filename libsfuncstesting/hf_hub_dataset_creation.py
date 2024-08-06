from datasets import load_from_disk, Dataset

octopus_dataset = load_from_disk("/proj/mounted/datasets/octopus-non-overlapping")

def modify_octopus_data(example):
    example["api_name"] = example["response"].split('(')[0]
    return example

octopus_dataset["train"] = octopus_dataset["train"].map(modify_octopus_data)
octopus_dataset["test"] = octopus_dataset["test"].map(modify_octopus_data)

def modify_octopus_apis(example):
    start_pos = example["api_description"].find("def ") + len("def ")
    end_pos = example["api_description"].find("(", start_pos)
    example["api_name"] = example["api_description"][start_pos:end_pos]
    return example

with open("/proj/mounted/func-docs/documentation-octopus.txt", "r") as f:
    func_text = f.read()
octopus_api_list = func_text.split("\n\n\n")
octopus_api_dataset = Dataset.from_dict({"api_description": octopus_api_list})
octopus_dataset["APIs"] = octopus_api_dataset
octopus_dataset["APIs"] = octopus_dataset["APIs"].map(modify_octopus_apis)

toole_dataset = load_from_disk("/proj/mounted/datasets/toole-single-tool-non-overlapping")

def modify_toole_data(example):
    example["api_name"] = example["response"]
    return example

toole_dataset["train"] = toole_dataset["train"].map(modify_toole_data)
toole_dataset["test"] = toole_dataset["test"].map(modify_toole_data)

def modify_toole_apis(example):
    example["api_name"] = example["api_description"].split(':')[0]
    return example

with open("/proj/mounted/func-docs/documentation-toole.txt", "r") as f:
    func_text = f.read()
toole_api_list = func_text.split("\n")

toole_api_dataset = Dataset.from_dict({"api_description": toole_api_list})
toole_dataset["APIs"] = toole_api_dataset
toole_dataset["APIs"] = toole_dataset["APIs"].map(modify_toole_apis)

toole_dataset.save_to_disk("/proj/mounted/to_the_hub_datasets/ToolENonOverlapping")
octopus_dataset.save_to_disk("/proj/mounted/to_the_hub_datasets/OctopusNonOverlapping")
