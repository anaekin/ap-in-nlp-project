# Install packages if required
# pip install torch transformers dataset evaluate nltk accelerate absl-py rouge_score bert_score bitsandbytes

# For RuntimeError: NCCL Error 2: unhandled system error
# export CUDA_VISIBLE_DEVICES=<gpu-id>
# <gpu-id> depends on the GPU you want to use
# Ex - export CUDA_VISIBLE_DEVICES=0

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)
from evaluate import load
import os
import numpy as np
import gc
import torch
import json
import nltk

nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# from bitsandbytes.optim import AdamW8bit  # If using 8bit optimizer

# Set CUDA configurations for training and parallelism
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the ROUGE metric
all_metrics = ["rouge", "bertscore", "bleu", "meteor"]
loaded_metrics = {metric: load(metric) for metric in all_metrics}


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


def tokenize_dataset(
    dataset,
    tokenizer,
    params,
):
    max_source_length = params["max_source_length"]
    max_target_length = params["max_target_length"]
    prefix = params["prefix"]

    def tokenize_function(examples):
        inputs = [prefix + doc for doc in examples["dialogue"]]
        summaries = examples["summary"]

        # Tokenize inputs and targets (summaries)
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )

        model_target = tokenizer(
            summaries,
            max_length=max_target_length,
            truncation=True,
        )

        # Directly assign labels to model inputs
        model_inputs["labels"] = model_target["input_ids"]

        return model_inputs

    # Tokenizing and returning dataset with optimized structure
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["dialogue", "summary", "topic"],  # Remove original text columns
    )


def preprocess_and_tokenize_dataset(
    tokenizer,
    params,
):
    dataset_name = params["dataset_name"]
    dataset_save_path = params["dataset_save_path"]

    if os.path.exists(dataset_save_path):
        print(f"Loading tokenized dataset from {dataset_save_path}...")
        tokenized_dataset = load_from_disk(dataset_save_path)

        print("Tokenized dataset loaded.")
        print("\n")
    else:
        dataset = load_dataset(dataset_name)

        print("Tokenizing dataset...")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, params)
        print("Dataset tokenization complete.")
        print("\n")

        print(f"Saving tokenized dataset to {dataset_save_path}...")
        tokenized_dataset.save_to_disk(dataset_save_path)

        print(f"Tokenized dataset saved to {dataset_save_path}.")
        print("\n")

    # Set the format to PyTorch
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset


def prepare_training_args(params, generation_config=None):
    print("##################### Hyperparameters #####################")
    print(json.dumps(params, indent=4))
    print("###########################################################")

    output_dir = params["model_training_output_path"]
    weight_decay = params["weight_decay"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    grad_accum_steps = params["grad_accum_steps"]
    num_train_epochs = params["num_train_epochs"]
    warmup_steps = params["warmup_steps"]
    eval_steps = params["eval_steps"]
    save_steps = params["save_steps"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        report_to="none",  # No logging to external tools
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=100,
        weight_decay=weight_decay,  # Helps prevent overfitting
        eval_strategy="steps",
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,  # Adjust based on dataset size
        save_strategy="steps",
        save_steps=save_steps,
        dataloader_drop_last=False,
        fp16=True,  # Enable mixed precision for memory efficiency
        predict_with_generate=True,  # Required for summarization tasks
        dataloader_num_workers=2,  # Prevent memory bottlenecks
        metric_for_best_model="rougeL",  # Use ROUGE-L for best model selection
        greater_is_better=True,  # Higher ROUGE-L is better
        load_best_model_at_end=True,  # Load the best model at the end of training
        lr_scheduler_type="linear",
        learning_rate=learning_rate,  # Standard for fine-tuning large models
        per_device_train_batch_size=batch_size,  # Maximize within GPU capacity
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # Simulates larger batch size
        num_train_epochs=num_train_epochs,
        generation_config=generation_config,
    )
    return training_args


def prepare_data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    return data_collator


def compute_metrics_function(eval_preds, tokenizer, metrics=all_metrics):
    predictions, labels = eval_preds
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [
        "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
    ]

    results = {}
    for metric_name in metrics:
        try:
            metric = loaded_metrics[metric_name]
            if metric_name == "rouge":
                metric_result = metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=True,
                )
                results.update(
                    {key: value * 100 for key, value in metric_result.items()}
                )
            elif metric_name == "bertscore":
                metric_result = metric.compute(
                    predictions=decoded_preds, references=decoded_labels, lang="en"
                )  # Fixed: Removed string conversion
                results.update(
                    {
                        f"bert_{key}": np.mean(value)
                        for key, value in metric_result.items()
                    }
                )  # Improved key naming
            else:  # bleu, meteor
                metric_result = metric.compute(
                    predictions=decoded_preds, references=decoded_labels
                )
                results.update(
                    {
                        key: value * 100
                        for key, value in metric_result.items()
                        if key != "precisions"
                    }
                )

        except Exception as e:
            print(
                f"Error computing {metric_name}: {e}"
            )  # More informative error message

    # Add mean generated and reference lengths
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    reference_lens = [
        np.count_nonzero(label != tokenizer.pad_token_id) for label in labels
    ]
    results["gen_len"] = np.mean(prediction_lens)
    results["avg_ref_len"] = np.mean(reference_lens)

    return {k: round(v, 4) for k, v in results.items()}


def create_trainer(
    model,
    training_args,
    preprocessed_dataset,
    tokenizer,
    metrics=all_metrics,
):
    data_collator = prepare_data_collator(tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset["train"],
        eval_dataset=preprocessed_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=(
            (
                lambda eval_preds: compute_metrics_function(
                    eval_preds, tokenizer, metrics
                )
            )
            if len(metrics) > 0
            else None
        ),
    )
    return trainer


def save_model(model, tokenizer, params):
    print("Saving the model...")
    model_save_path = params["model_save_path"]
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}.")


def save_params(params):
    # Save the parameters
    with open(params["params_save_path"], "w") as f:
        json.dump(params, f, indent=4)
    print(f"Training parameters saved to {params['params_save_path']}.")
    print("\n")


def add_steps_params(params, total_train_examples):
    params["total_train_examples"] = total_train_examples
    steps_per_epoch = total_train_examples // (
        params["batch_size"] * params["grad_accum_steps"]
    )  # 12600 / (16 * 4) = 196.875

    # Calculate total steps by multiplying steps per epoch by the number of epochs
    total_steps = int(
        steps_per_epoch * params["num_train_epochs"]
    )  # 196.875 * 10 = ~1968

    eval_steps = int(
        total_steps * params["eval_steps_factor"]
    )  # total_steps * 0.2 = ~393.6

    # Calculate warmup steps as a fraction of total steps
    warmup_steps = int(
        total_steps * params["warmup_steps_factor"]
    )  # total_steps * 0.1 = ~196.8

    params["total_steps"] = total_steps
    params["warmup_steps"] = warmup_steps
    params["eval_steps"] = eval_steps
    params["save_steps"] = int(eval_steps * params["save_steps_multiple"])

    return params


def fine_tune_model(params, metrics=None):
    model_checkpoint = params["model_checkpoint"]
    device = check_device()

    # Preprocessing #####################
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preprocess the dataset
    tokenized_dataset = preprocess_and_tokenize_dataset(tokenizer, params)

    # Automatically generate steps related params like eval_steps, warmup_steps, save_steps
    total_train_examples = len(tokenized_dataset["train"])
    params = add_steps_params(params, total_train_examples)

    # Training #########################
    # Load the model
    print(f"Loading the model {model_checkpoint}...")

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.to(device)
    print(f"Model loaded on device: {model.device}.")
    print("\n")

    model_generation_config = None
    if params["generation_config"] is not None:
        model_generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            model_checkpoint, **params["generation_config"], return_unused_kwargs=True
        )
        # params["unused_kwargs"] = unused_kwargs
        params["generation_config"] = model_generation_config.to_dict()

        print("Generation config:", model_generation_config)
        print("Unused kawargs:", unused_kwargs)
        print("\n")

    # Prepare the training arguments
    training_args = prepare_training_args(params, model_generation_config)
    save_params(params)

    # Create a model trainer
    trainer = create_trainer(
        model,
        training_args=training_args,
        preprocessed_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        metrics=metrics,
    )

    try:
        # Train the model
        trainer.train()

        print("Model training complete.")
        print("\n")

        # Save the model
        save_model(
            model,
            tokenizer,
            params,
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM exception encountered.")
        print("\n")
    finally:
        print("##################### Hyperparameters #####################")
        print(json.dumps(params, indent=4))
        print("###########################################################")
        clear_cache(model, tokenizer, trainer)


################################################################################
if __name__ == "__main__":
    output_folder = "output"
    params = {
        "model_checkpoint": "facebook/bart-large-xsum",
        "dataset_name": "knkarthick/dialogsum",
        # Save folder paths
        "model_training_output_path": f"./{output_folder}/training-output",
        "model_save_path": f"./{output_folder}/finetuned-output",
        "dataset_save_path": f"./{output_folder}/dataset-preprocessing-output",
        "params_save_path": f"./{output_folder}/training_params.json",
    }

    # For Jupyter Lab - Best (Less performant but more accurate)
    # Example calculation
    # steps_per_epoch = total_train_examples / (batch_size * grad_accum_steps) = 12600 / (16 * 4) = 196.875
    # total_steps = steps_per_epoch * num_train_epochs = 196.875 * 10 = ~1968
    # eval_steps = total_steps * eval_steps_factor = 1968 * 0.1 = ~196
    # warmup_steps = total_steps * warmup_steps_factor = 1968 * 0.2 = ~393
    # save_steps = eval_steps * save_steps_multiple = 196 * 3 = 588
    params["prefix"] = (
        "Generate a summary for the following dialogue: "  # In case of T5 model, use 'summarize: '
    )
    params["max_source_length"] = (
        512  # Check dataset-analysis.ipynb for max_source_length
    )
    params["max_target_length"] = (
        90  # Check dataset-analysis.ipynb for max_target_length
    )
    params["weight_decay"] = 0.01
    params["learning_rate"] = 5e-5
    params["batch_size"] = 30
    params["grad_accum_steps"] = 4
    params["num_train_epochs"] = 6
    params["save_steps_multiple"] = (
        5  # If 5, the model will save after every 5 * eval_steps
    )
    params["warmup_steps_factor"] = (
        0.1  # If 0.1, the model will warmup for 0.1 * total_steps
    )
    params["eval_steps_factor"] = (
        0.1  # If 0.2, the model will eval after every 0.2 * steps_per_epoch
    )

    # For mobile GPU (RTX 3070 Ti) - Balanced Performance and Accuracy - Takes ~6 hours
    # params["learning_rate"] = 3e-5
    # params["batch_size"] = 2
    # params["grad_accum_steps"] = 2
    # params["num_train_epochs"] = 3

    # params["generation_config"] = None
    params["generation_config"] = {
        "max_length": params["max_target_length"],
        "num_beams": 6,
        # "top_k": 50,
        # "top_p": 0.9,
        # "temperature": 0.8,
        "no_repeat_ngram_size": 3,
        "length_penalty": 2,
        "do_sample": True,
        "_from_model_config": False,
    }

    # Fine-tuning the model ##############
    fine_tune_model(params, metrics=["rouge", "bleu"])
