from datasets import load_dataset

dataset = load_dataset("knkarthick/dialogsum")


def reduce_dataset_sizes(dataset, num_samples_per_split=200):
    for (
        split_name
    ) in dataset.keys():  # Iterate through all splits ('train', 'validation', 'test')

        if split_name == "train":
            reduced_split = dataset[split_name].shuffle(seed=42).select(range(20))

        if split_name == "train":
            reduced_split = dataset[split_name].shuffle(seed=42).select(range(20))

        if split_name == "train":
            reduced_split = dataset[split_name].shuffle(seed=42).select(range(20))

        dataset[split_name] = reduced_split
    return dataset


dataset = reduce_dataset_sizes(dataset)
