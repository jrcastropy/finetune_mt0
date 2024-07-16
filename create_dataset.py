from datasets import DatasetDict, Dataset

import glob, json, random

def create_dataset(ds_path, file_type="json", train_test_split=0.20, ds_percentage=1):
    datasets = glob.glob(f'{ds_path}/*.{file_type}')

    test_ds = []
    train_ds = []

    lowest_dataset_samples = 0
    for ds_fn in datasets:
        with open(ds_fn, 'r') as f:
            synth_dataset = json.loads(f.read())
            if len(synth_dataset) > lowest_dataset_samples:
                lowest_dataset_samples = len(synth_dataset)
    per_dataset_samples = int((lowest_dataset_samples * len(datasets)) * ds_percentage)

    print("Total Dataset Count:", (lowest_dataset_samples * len(datasets)))
    print("Lowest Dataset Sample:", lowest_dataset_samples)
    print("DS per samples:", per_dataset_samples)
    print("Total datasets after reduce:", per_dataset_samples*len(datasets))
    print()
    
    for ds_fn in datasets:
        with open(ds_fn, 'r') as f:
            synth_dataset = json.loads(f.read())
        new_synth_dataset = []
        random.shuffle(synth_dataset)
        synth_dataset = synth_dataset[:per_dataset_samples]
        print(ds_fn, "- count:", len(synth_dataset))
        for x in synth_dataset:
            query = x['query']
            extracted_data = x['extracted_data']
            if extracted_data['tags'] is None:
                extracted_data['tags'] = []
            if extracted_data['cuisines'] is None:
                extracted_data['cuisines'] = []
            new_synth_dataset.append({"query": query, "extracted_data": json.dumps(extracted_data, ensure_ascii=False)})
        rn_idx = random.choice(range(len(new_synth_dataset)))
        print("Sample Data:", new_synth_dataset[rn_idx])

        test_ds_c = round(len(new_synth_dataset) * train_test_split)
        test_ds.extend(new_synth_dataset[:test_ds_c])
        train_ds.extend(new_synth_dataset[test_ds_c:])
        print("*"*100, "\n")
    
    dataset_dict = DatasetDict()

    dataset_dict['train'] = Dataset.from_list(train_ds)
    dataset_dict['test'] = Dataset.from_list(test_ds)

    return dataset_dict