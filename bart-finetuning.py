# Install packages if required
# %pip install torch transformers dataset evaluate nltk accelerate absl-py rouge_score

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from evaluate import load
import os
import numpy as np
import gc
import torch
import nltk

nltk.download("punkt_tab")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the ROUGE metric
metric = load("rouge")


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def clear_cache(model):
    print("Freeing CUDA memory...")
    del model  # Delete the model object
    gc.collect()  # Collect unused objects
    torch.cuda.empty_cache()  # Clear the CUDA cache
    print("CUDA cache cleared and GPU memory released.")


def load_and_preprocess_data(
    tokenizer,
    dataset,
    max_source_length=512,
    max_target_length=128,
    prefix="",  # In case of T5 model
):
    dataset_name = dataset.split("/")[1]
    saved_dataset_path = f"./{dataset_name}_tokenized_{max_source_length}"
    dataset = load_dataset(dataset)

    # Load separately for evaluation
    original_test_dataset = dataset["test"]

    if os.path.exists(saved_dataset_path):
        print(f"Loading tokenized dataset from {saved_dataset_path}")
        tokenized_dataset = load_from_disk(saved_dataset_path)
    else:
        print("Preprocessing dataset...")

        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["dialogue"]]
            model_inputs = tokenizer(
                inputs, max_length=max_source_length, truncation=True
            )

            # Setup the tokenizer for targets
            labels = tokenizer(
                text_target=examples["summary"],
                max_length=max_target_length,
                truncation=True,
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
        )

        print(f"Saving to {saved_dataset_path}")
        tokenized_dataset.save_to_disk(saved_dataset_path)

    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]
    return train_dataset, val_dataset, test_dataset, original_test_dataset


def prepare_training_args(
    model_name,
    learning_rate=2e-5,
    batch_size=2,
    grad_accum_steps=4,
):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{model_name}-finetuned",
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        num_train_epochs=3,
        weight_decay=0.01,  # Helps prevent overfitting
        warmup_steps=500,  # Adjust based on dataset size
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        fp16=True,  # Enable mixed precision for memory efficiency
        predict_with_generate=True,  # Required for summarization tasks
        dataloader_num_workers=2,  # Prevent memory bottlenecks
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="none",  # No logging to external tools
        learning_rate=learning_rate,  # Standard for fine-tuning large models
        per_device_train_batch_size=batch_size,  # Maximize within GPU capacity
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # Simulates larger batch size
    )
    return training_args


def prepare_data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, return_tensors="pt", pad_to_multiple_of=128
    )
    return data_collator


def compute_metrics(eval_pred, tokenizer, metric):
    predictions, labels = eval_pred
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

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def create_trainer(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    training_args,
    metric,
):
    data_collator = prepare_data_collator(tokenizer, model)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, metric),
    )
    return trainer


def save_model(model, tokenizer, model_name):
    model_path = f"./{model_name}-finetuned-dialogsum"
    print(f"Saving the model to {model_path}...")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}.")


def fine_tune_model(model_checkpoint, dataset, device):
    try:
        model_name = model_checkpoint.split("/")[-1]

        # Preprocessing #####################
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Preprocess the dataset
        train_dataset, val_dataset, test_dataset, original_test_dataset = (
            load_and_preprocess_data(
                tokenizer,
                dataset,
                max_source_length=max_source_length,
                max_target_length=max_target_length,
            )
        )

        # Load the model
        print(f"Loading the model {model_checkpoint}...")

        config = AutoConfig.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_config(config)
        model.to(device)

        print("Model loaded successfully.")
        print(f"Model is on: {model.device}")

        # Prepare the training arguments
        training_args = prepare_training_args(
            model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )

        # Create a model trainer
        trainer = create_trainer(
            model,
            train_dataset,
            val_dataset,
            tokenizer,
            training_args,
            metric,
        )

        # Train the model
        trainer.train()

        print("Model training complete.")

        # Save the model
        save_model(model, tokenizer, model_name)
    except torch.cuda.OutOfMemoryError:
        print("OOM exception encountered.")
    finally:
        clear_cache(model)


if __name__ == "__main__":
    ####################################
    device = check_device()
    model = "facebook/bart-large-cnn"
    dataset = "knkarthick/dialogsum"

    # For mobile GPU (RTX 3070 Ti) - Takes ~5 hours 30 minutes
    # max_source_length = 512
    # max_target_length = 128
    # learning_rate = 2e-5
    # batch_size = 2
    # grad_accum_steps = 4

    # For mobile GPU (RTX 3070 Ti) - Balanced Performance and Accuracy - Takes ~6 hours
    # max_source_length = 1024
    # max_target_length = 128
    # learning_rate = 2e-5
    # batch_size = 1
    # grad_accum_steps = 4

    # For Jupyter Lab - Balanced Performance and Accuracy
    # max_source_length = 1024
    # max_target_length = 128
    # learning_rate = 2e-5
    # batch_size = 2
    # grad_accum_steps = 4

    # For Jupyter Lab - Best (Less performant but more accurate)
    max_source_length = 1429
    max_target_length = 249
    learning_rate = 2e-5
    batch_size = 1
    grad_accum_steps = 8

    print("##################### Hyperparameters #####################")
    print("Model:", model)
    print("Dataset:", dataset)
    print("Max Source Length:", max_source_length)
    print("Max Target Length:", max_target_length)
    print("Learning Rate:", learning_rate)
    print("Batch Size:", batch_size)
    print("Grad Accum Steps:", grad_accum_steps)
    print("Device:", device)
    print("###########################################################")

    # Fine-tuning the model ##############
    fine_tune_model(model, dataset, device)
