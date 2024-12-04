# Install packages if required
# pip install torch transformers dataset evaluate nltk accelerate absl-py rouge_score bitsandbytes
# For RuntimeError: NCCL Error 2: unhandled system error
# export CUDA_VISIBLE_DEVICES=0

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
import re
import json
from bitsandbytes.optim import AdamW8bit  # If using 8bit optimizer

nltk.download("punkt_tab")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the ROUGE metric
metric = load("rouge")


def clear_cache(model, tokenizer, trainer):
    print("Freeing CUDA memory...")
    del model  # Delete the model object
    del tokenizer  # Delete the tokenizer object
    del trainer
    torch.cuda.empty_cache()  # Clear the CUDA cache
    gc.collect()  # Collect unused objects
    print("CUDA cache cleared and GPU memory released.")
    print("\n")


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def clean_dialogue(dialogue):
    # Remove unnecessary whitespace
    dialogue = re.sub(r"\s+", " ", dialogue).strip()

    # Standardize speaker tags
    dialogue = re.sub(
        r"#Person\d+#", lambda match: f"Speaker {match.group()[7]}", dialogue
    )

    # Normalize punctuation (e.g., replace ellipses with single period)
    dialogue = dialogue.replace("...", ".")

    # Remove unwanted special characters
    dialogue = re.sub(r"[^\w\s.,!?:']", "", dialogue)

    return dialogue


def tokenize_dataset(
    dataset, tokenizer, max_source_length=512, max_target_length=128, prefix=""
):
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["dialogue"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess_function,
        batched=True,
    )


def load_and_preprocess_data(
    tokenizer,
    dataset,
    max_source_length=512,
    max_target_length=128,
    prefix="",  # In case of T5 model
):
    dataset_name = dataset.split("/")[1]
    saved_dataset_path = (
        f"./{dataset_name}_tokenized_{max_source_length}_{max_target_length}"
    )

    uncleaned_dataset = load_dataset(dataset)

    print("Cleaning dataset...")
    cleaned_dataset = uncleaned_dataset.map(
        lambda x: {"dialogue": clean_dialogue(x["dialogue"])}
    )

    print("Cleaning complete.")
    print("\n")

    # Example Output
    print("Training data (after cleaning):", cleaned_dataset["train"][0])
    print("\n")

    # Load separately for evaluation
    original_test_dataset = cleaned_dataset["test"]

    if os.path.exists(saved_dataset_path):
        print(f"Loading tokenized dataset from {saved_dataset_path}...")
        tokenized_dataset = load_from_disk(saved_dataset_path)

        print("Tokenized dataset loaded.")
        print("\n")
    else:

        print("Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(
            cleaned_dataset, tokenizer, max_source_length, max_target_length, prefix
        )
        print("Dataset tokenization complete.")
        print("\n")

        print(f"Saving tokenized dataset to {saved_dataset_path}...")
        tokenized_dataset.save_to_disk(saved_dataset_path)

        print(f"Tokenized dataset saved to {saved_dataset_path}.")
        print("\n")

    # Set the format to PyTorch
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    return train_dataset, val_dataset, test_dataset, original_test_dataset


def prepare_training_args(
    output_dir,
    learning_rate=2e-5,
    batch_size=2,
    grad_accum_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    optim="adamw_8bit",
):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        weight_decay=weight_decay,  # Helps prevent overfitting
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
        optim=optim,  # if using 8-bit AdamW
        learning_rate=learning_rate,  # Standard for fine-tuning large models
        per_device_train_batch_size=batch_size,  # Maximize within GPU capacity
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # Simulates larger batch size
        num_train_epochs=num_train_epochs,
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
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Directly use the decoded predictions and labels without sentence tokenization
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )

    # Convert to percentage
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

    if training_args.optim == "adamw_8bit":
        optimizer = AdamW8bit(
            model.parameters(), lr=training_args.learning_rate
        )  # Create optimizer instance if necessary

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, metric),
        optimizers=(
            (optimizer, None) if training_args.optim == "adamw_8bit" else None
        ),  # Pass to optimizers if using 8bit adam
    )
    return trainer


def save_model(model_save_path, model, tokenizer, params):
    print(f"Saving the model...")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}.")

    # Save the parameters
    params_path = os.path.join(model_save_path, "training_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Training parameters saved to {params_path}.")
    print("\n")


def fine_tune_model(params):
    try:
        model_checkpoint = params["model_checkpoint"]
        dataset = params["dataset"]

        model_name = model_checkpoint.split("/")[-1]
        training_output_dir = f"./{model_name}-training-output"
        model_save_path = f"./{model_name}-finetuned-dialogsum"

        device = check_device()

        # Preprocessing #####################
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Preprocess the dataset
        train_dataset, val_dataset, test_dataset, original_test_dataset = (
            load_and_preprocess_data(
                tokenizer,
                dataset,
                max_source_length=params["max_source_length"],
                max_target_length=params["max_target_length"],
            )
        )

        # Training #########################
        # Load the model
        print(f"Loading the model {model_checkpoint}...")

        config = AutoConfig.from_pretrained(model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_config(config)
        model.to(device)

        print(f"Current device: {torch.cuda.current_device()}.")
        print(f"Model loaded on device: {model.device}.")
        print("\n")

        # Prepare the training arguments
        training_args = prepare_training_args(
            output_dir=training_output_dir,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            grad_accum_steps=params["grad_accum_steps"],
            num_train_epochs=params["num_train_epochs"],
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
        print("\n")

        # Save the model
        save_model(
            model_save_path,
            model,
            tokenizer,
            params,
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM exception encountered.")
        print("\n")
    finally:
        clear_cache(model, tokenizer, trainer)


if __name__ == "__main__":
    ####################################
    params = {
        "model_checkpoint": "facebook/bart-large-xsum",
        "dataset": "knkarthick/dialogsum",
    }

    # For mobile GPU (RTX 3070 Ti) - Takes ~5 hours 30 minutes
    # params["max_source_length"] = 512
    # params["max_target_length"] = 128
    # params["learning_rate"] = 2e-5
    # params["batch_size"] = 2
    # params["grad_accum_steps"] = 4
    # params["num_train_epochs"] = 3

    # For mobile GPU (RTX 3070 Ti) - Balanced Performance and Accuracy - Takes ~6 hours
    # params["max_source_length"] = 1024
    # params["max_target_length"] = 128
    # params["learning_rate"] = 2e-5
    # params["batch_size"] = 1
    # params["grad_accum_steps"] = 4
    # params["num_train_epochs"] = 3

    # For Jupyter Lab - Balanced Performance and Accuracy
    # params["max_source_length"] = 1024
    # params["max_target_length"] = 128
    # params["learning_rate"] = 2e-5
    # params["batch_size"] = 2
    # params["grad_accum_steps"] = 4
    # params["num_train_epochs"] = 3

    # For Jupyter Lab - Best (Less performant but more accurate)
    params["max_source_length"] = 1024
    params["max_target_length"] = 200
    params["learning_rate"] = 3e-5
    params["batch_size"] = 4
    params["grad_accum_steps"] = 4
    params["num_train_epochs"] = 15

    print("##################### Hyperparameters #####################")
    print(json.dumps(params, indent=4))
    print("###########################################################")

    # Fine-tuning the model ##############
    fine_tune_model(params)

    print("##################### Hyperparameters #####################")
    print(json.dumps(params, indent=4))
    print("###########################################################")
