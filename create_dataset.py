from datasets import DatasetDict, Dataset

import glob, json, random

def create_dataset(ds_path):
    datasets = glob.glob(f'{ds_path}/*.json')

    test_ds = []
    train_ds = []

    for ds_fn in datasets:
        print(ds_fn)
        with open(ds_fn, 'r') as f:
            ds_data = f.read()
            synth_dataset = json.loads(ds_data)
            print(len(synth_dataset))
        # synth_dataset = [{"query": x['query'], "extracted_data": json.dumps(x['extracted_data'], ensure_ascii=False)} for x in synth_dataset]
        new_synth_dataset = []
        for x in synth_dataset:
            query = x['query']
            extracted_data = x['extracted_data']
            if extracted_data['tags'] is None:
                extracted_data['tags'] = []
            if extracted_data['cuisines'] is None:
                extracted_data['cuisines'] = []
            new_synth_dataset.append({"query": query, "extracted_data": json.dumps(extracted_data, ensure_ascii=False)})

        print("Sample Data:", new_synth_dataset[0])

        test_ds_c = round(len(new_synth_dataset) * 0.20)
        test_ds.extend(new_synth_dataset[:test_ds_c])
        train_ds.extend(new_synth_dataset[test_ds_c:])
        # type(synth_dataset), len(synth_dataset)
        print("*"*100, "\n")

    random.shuffle(train_ds)
    random.shuffle(test_ds)

    dataset_dict = DatasetDict()

    dataset_dict['train'] = Dataset.from_list(train_ds)
    dataset_dict['test'] = Dataset.from_list(test_ds)

    return dataset_dict