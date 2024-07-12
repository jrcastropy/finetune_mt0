from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import DatasetDict, Dataset, concatenate_datasets
from create_dataset import create_dataset
from nltk.tokenize import sent_tokenize
from huggingface_hub import HfFolder
from functools import partial

import json_repair, nltk, evaluate, random, json, pickle
import numpy as np

def tokenize_data(tokenizer, dataset_dict, tokenization_type="inputs"):

    if tokenization_type == "inputs":
        # The maximum total input sequence length after tokenization.
        # Sequences longer than this will be truncated, sequences shorter will be padded.
        tokenized_inputs = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]]).map(lambda x: tokenizer(x["query"], truncation=True), batched=True, remove_columns=["query", "extracted_data"])
        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        print(f"Max source length: {max_source_length}")
        return tokenized_inputs, max_source_length

    elif tokenization_type == "targets":
        # The maximum total sequence length for target text after tokenization.
        # Sequences longer than this will be truncated, sequences shorter will be padded."
        tokenized_targets = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]]).map(lambda x: tokenizer(x["extracted_data"], truncation=True), batched=True, remove_columns=["query", "extracted_data"])
        max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
        print(f"Max target length: {max_target_length}")
        return tokenized_targets, max_target_length

    return False, False

def preprocess_function(sample, tokenizer, max_source_length, max_target_length, padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["query"]]
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length*3, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["extracted_data"], max_length=max_target_length*3, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def create_trainer(repository_id, 
        model, 
        tokenizer, 
        tokenized_dataset, 
        partial_compute_metrics,
        # we want to ignore tokenizer pad token in the loss
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        # training args
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=3,
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save"
    ):

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of
    )

    logging_dir=f"{repository_id}/logs"

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=predict_with_generate,
        fp16=fp16,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        report_to=report_to,
        push_to_hub=push_to_hub,
        hub_strategy=hub_strategy,
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token()
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=partial_compute_metrics,
    )

    return trainer

def save_data_pickle(data, filename):
    with open(f'{filename}.pkl', 'wb') as f:  # open a text file
        pickle.dump(data, f) # serialize the list

def load_data_pickle(filename):
    with open(f'{filename}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data