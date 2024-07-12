from utils import *

def main(model_id, repository_id, metric, ds_path, load_data_only=True, pickle_path="pickle_data"):

    # Load tokenizer of MT0-Small
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_tokens(['{', '}', "[", "]"])

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')
    model.resize_token_embeddings(len(tokenizer))
    
    if load_data_only:
        dataset_dict = load_data_pickle(f"{pickle_path}/dataset_dict")

        tokenized_inputs = load_data_pickle(f"{pickle_path}/tokenized_inputs")
        max_source_length = load_data_pickle(f"{pickle_path}/max_source_length")
        tokenized_targets = load_data_pickle(f"{pickle_path}/tokenized_targets")
        max_target_length = load_data_pickle(f"{pickle_path}/max_target_length")

        tokenized_dataset = load_data_pickle(f"{pickle_path}/tokenized_dataset")
    else:
        dataset_dict = create_dataset(ds_path)
        save_data_pickle(dataset_dict, f"{pickle_path}/dataset_dict")

        tokenized_inputs, max_source_length = tokenize_data(tokenizer, dataset_dict, tokenization_type="inputs")
        tokenized_targets, max_target_length = tokenize_data(tokenizer, dataset_dict, tokenization_type="targets")

        save_data_pickle(tokenized_inputs, f"{pickle_path}/tokenized_inputs")
        save_data_pickle(max_source_length, f"{pickle_path}/max_source_length")
        save_data_pickle(tokenized_targets, f"{pickle_path}/tokenized_targets")
        save_data_pickle(max_target_length, f"{pickle_path}/max_target_length")

        preprocess_function_partial = partial(preprocess_function, tokenizer=tokenizer, max_source_length=max_source_length, max_target_length=max_target_length)
        tokenized_dataset = dataset_dict.map(preprocess_function_partial, batched=True, remove_columns=["query", "extracted_data"])
        save_data_pickle(tokenized_dataset, f"{pickle_path}/tokenized_dataset")
        print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    
    partial_compute_metrics = partial(compute_metrics, tokenizer=tokenizer, metric=metric)
    trainer = create_trainer(repository_id, model, tokenizer, tokenized_dataset, partial_compute_metrics)

    # Start training
    trainer.train()
    trainer.evaluate()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(repository_id)
    trainer.create_model_card()
    # Push the results to the hub

    trainer.push_to_hub()


if __name__ == "__main__":
    repository_id = f"mt0-small-query-extraction-v4"
    model_id = "bigscience/mt0-small"
    metric = evaluate.load("rouge")
    ds_path = "datasets"
    main(model_id, repository_id, metric, ds_path)