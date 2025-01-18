import json
from datasets import Dataset, DatasetDict
from evaluate import load
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_checkpoint = "facebook/bart-large"
metric = load("rouge")

TEST_SUMMARY_ID = 1


def transform_single_dialogsumm_file(file):
    data = open(file, "r").readlines()
    result = {"fname": [], "summary": [], "dialogue": []}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
    return Dataset.from_dict(result)


def transform_test_file(file):
    data = open(file, "r").readlines()
    result = {"fname": [], "summary%d" % TEST_SUMMARY_ID: [], "dialogue": []}
    for i in data:
        d = json.loads(i)
        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])

    result["summary"] = result["summary%d" % TEST_SUMMARY_ID]
    return Dataset.from_dict(result)


def transform_dialogsumm_to_huggingface_dataset(train, validation, test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_test_file(test)
    return DatasetDict({"train": train, "validation": validation, "test": test})


raw_datasets = transform_dialogsumm_to_huggingface_dataset(
    "./DialogSum_Data/dialogsum.train.jsonl",
    "./DialogSum_Data/dialogsum.dev.jsonl",
    "./DialogSum_Data/dialogsum.test.jsonl",
)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, dropout=0.1)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 256
max_target_length = 128


def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_bart_dialogsum/output",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    metric_for_best_model="eval_rouge1",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=15,
    learning_rate=3e-5,
    lr_scheduler_type="polynomial",
    warmup_steps=200,
    weight_decay=0.01,
    max_grad_norm=0.1,
    fp16=True,
    save_total_limit=1,
    label_smoothing_factor=0.1,
    dataloader_num_workers=8,
    logging_dir="./finetuned_bart_dialogsum/logs",
    predict_with_generate=True,
    generation_max_length=100,
    generation_num_beams=5,
    seed=42,
    load_best_model_at_end=True,
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import nltk
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    results = {}

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results\
    results.update({key: value * 100 for key, value in result.items()})

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
)


trainer.train()


def save_model(model, tokenizer, model_save_path):
    print("Saving the model...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}.")


save_model(model, tokenizer, model_save_path="./finetuned_bart_dialogsum")


print("#" * 10)
print("Evaluating on test set...")

out = trainer.predict(tokenized_datasets["test"], num_beams=5)

predictions, labels, metric = out
print(metric)
