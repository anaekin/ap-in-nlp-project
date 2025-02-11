# Install packages if required
# pip install torch transformers dataset evaluate nltk accelerate absl-py rouge_score bert_score wandb

# For RuntimeError: NCCL Error 2: unhandled system error
# export CUDA_VISIBLE_DEVICES=<gpu-id>
# <gpu-id> depends on the GPU you want to use
# Ex - export CUDA_VISIBLE_DEVICES=0

import wandb

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback,
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

WANDB_PROJECT_NAME = "ap-in-nlp-project-bart"

# Set CUDA configurations for training and parallelism
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the ROUGE metric
all_metrics = ["rouge", "bertscore", "bleu", "meteor"]
loaded_metrics = {metric: load(metric) for metric in all_metrics}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def reduce_dataset_sizes(dataset, train_split=100, val_split=20, test_split=30):
    for (
        split_name
    ) in dataset.keys():  # Iterate through all splits ('train', 'validation', 'test')

        if split_name == "train":
            reduced_split = (
                dataset[split_name].shuffle(seed=42).select(range(train_split))
            )

        if split_name == "validation":
            reduced_split = (
                dataset[split_name].shuffle(seed=42).select(range(val_split))
            )

        if split_name == "test":
            reduced_split = (
                dataset[split_name].shuffle(seed=42).select(range(test_split))
            )

        dataset[split_name] = reduced_split
    return dataset


def clear_cache(model, tokenizer, dataset):
    print("Freeing CUDA memory...")
    del model  # Delete the model object
    del tokenizer  # Delete the tokenizer object
    del dataset
    torch.cuda.empty_cache()  # Clear the CUDA cache
    gc.collect()  # Collect unused objects
    print("CUDA cache cleared and GPU memory released.")
    print("\n")


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

        if params["use_small_dataset"]:
            dataset = reduce_dataset_sizes(dataset)

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
    print("#" * 15 + " Hyperparameters " + "#" * 25)
    print(json.dumps(params, indent=4))
    print("#" * 45)

    training_args = Seq2SeqTrainingArguments(
        output_dir=params["output_dir"],
        save_total_limit=3,
        report_to="wandb",
        dataloader_num_workers=2,  # Prevent memory bottlenecks
        fp16=True,  # Enable mixed precision for memory efficiency
        logging_dir=params["logging_dir"],
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        gradient_checkpointing=True,
        logging_steps=params["logging_steps"],
        predict_with_generate=True,  # Required for summarization tasks
        metric_for_best_model=params[
            "metric_for_best_model"
        ],  # Use ROUGE-1 for best model selection
        load_best_model_at_end=True,  # Load the best model at the end of training
        weight_decay=params["weight_decay"],  # Helps prevent overfitting
        eval_steps=params["eval_steps"],  # Adjust based on dataset size
        warmup_ratio=params["warmup_ratio"],  # Adjust based on dataset size
        save_steps=params["save_steps"],  # Adjust based on dataset size
        lr_scheduler_type=params[
            "lr_scheduler_type"
        ],  # linear or cosine, tre different lr schedulers
        learning_rate=params[
            "learning_rate"
        ],  # Standard for fine-tuning large models, 5e-05
        per_device_train_batch_size=params["per_device_train_batch_size"],  # Default 8
        per_device_eval_batch_size=params["per_device_train_batch_size"],  # Default 8
        gradient_accumulation_steps=params[
            "gradient_accumulation_steps"
        ],  # Simulates larger batch size
        num_train_epochs=params[
            "num_train_epochs"
        ],  # Smaller batch size requires more epochs
        generation_config=generation_config,  # For generation during evaluation
        run_name=params["run_name"],
    )
    return training_args


def compute_metrics_function(eval_preds, tokenizer, params):
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

    print("decoded_preds: ", decoded_preds[0])
    print("\n")
    print("decoded_labels: ", decoded_labels[0])
    print("\n")

    metrics = params["metrics"]
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


def model_init_function(
    trial,
    params,
):
    model_checkpoint = params["model_checkpoint"]
    # Load the model
    print(f"Loading the model {model_checkpoint}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,
    )
    model.to(device)
    return model


def create_trainer(
    params,
    tokenizer,
    tokenized_dataset,
    model=None,
    generation_config=None,
):
    # Prepare the training arguments
    training_args = prepare_training_args(params, generation_config)

    # Prepare the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    callbacks = []
    if params["use_early_stopping"]:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=params["early_stopping_patience"]
            )
        )

    model_init = lambda trial: model_init_function(trial, params)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=(
            (lambda eval_preds: compute_metrics_function(eval_preds, tokenizer, params))
        ),
        model_init=None if model is not None else model_init,
    )
    return trainer


def save_model(model, tokenizer, params):
    print("Saving the model...")
    model_save_path = params["model_save_path"]
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved to {model_save_path}.")


def save_params(params):
    os.makedirs(os.path.dirname(params["params_save_path"]), exist_ok=True)
    # Save the parameters
    with open(params["params_save_path"], "w") as f:
        json.dump(params, f, indent=4)
    print(f"Training parameters saved to {params['params_save_path']}.")
    print("\n")


def hyperparameter_space(trial):
    return {
        "project": WANDB_PROJECT_NAME,
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "eval_loss"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-4,
            },
            "per_device_train_batch_size": {
                "values": [4, 8, 12, 16],
            },
            "gradient_accumulation_steps": {
                "values": [2, 4, 6, 8],
            },
            "num_train_epochs": {
                "values": [5, 10, 15],
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-2,
                "max": 1e-1,
            },
            "warmup_ratio": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.3,
            },
        },
    }


def fine_tune_model(params):
    model_checkpoint = params["model_checkpoint"]
    # Preprocessing #####################
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Preprocess the dataset
    tokenized_dataset = preprocess_and_tokenize_dataset(tokenizer, params)

    # Add additional parameters - total_train_examples, steps_per_epoch, total_steps
    params["total_train_examples"] = len(tokenized_dataset["train"])
    params["steps_per_epoch"] = params["total_train_examples"] // (
        params["per_device_train_batch_size"] * params["gradient_accumulation_steps"]
    )  # 12600 / (16 * 4) = 196.875

    # Calculate total steps by multiplying steps per epoch by the number of epochs
    params["total_steps"] = int(
        params["steps_per_epoch"] * params["num_train_epochs"]
    )  # 196.875 * 10 = ~1968

    # Training #########################
    # Load the generation config
    model_generation_config = None
    if params["use_generation_config"]:
        model_generation_config, _ = GenerationConfig.from_pretrained(
            model_checkpoint, **params["generation_config"], return_unused_kwargs=True
        )

    try:
        if params["n_trials"] is not None:
            sweeper = create_trainer(
                params,
                tokenizer=tokenizer,
                tokenized_dataset=tokenized_dataset,
                generation_config=model_generation_config,
            )

            best_sweep = sweeper.hyperparameter_search(
                direction="minimize",  # Minimize validation loss
                hp_space=hyperparameter_space,  # Hyperparameter search space
                backend="wandb",  # Use Weights & Biases for tracking
                n_trials=params["n_trials"],
                sweep_id=params["sweep_id"],
            )

            # Update params with the best hyperparameters
            params = {**params, **best_sweep.hyperparameters}
            save_params(params)
            wandb.teardown()

        best_run = wandb.init()
        params["run_name"] = best_run.name

        model = model_init_function(None, params)

        # Create a model trainer
        trainer = create_trainer(
            params,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            model=model,
            generation_config=model_generation_config,
        )
        # Train the model
        trainer.train()

        print("Model training complete.")
        print("\n")

        print("Evaluating fine-tuned model...")
        final_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        print("#" * 15 + " Final Results " + "#" * 25)
        print(json.dumps(final_results, indent=4))
        print("#" * 45)
        print("\n")

        # Save the model
        save_model(
            model,
            tokenizer,
            params,
        )
        wandb.finish()
        wandb.teardown()
    except torch.cuda.OutOfMemoryError:
        print("OOM exception encountered.")
        print("\n")
    finally:
        print("#" * 15 + " Final Hyperparameters " + "#" * 25)
        print(json.dumps(params, indent=4))
        print("#" * 45)
        print("\n")
        clear_cache(model, tokenizer, tokenized_dataset)


################################################################################
if __name__ == "__main__":
    print("\n")

    def get_params(
        model_checkpoint,
        training_output_dir,
        dataset_output_dir,
        use_small_dataset=False,
        n_trials=None,
        sweep_id=None,
    ):

        # For Jupyter Lab
        # Example calculation for steps
        ###############################
        # steps_per_epoch = total_train_examples / (batch_size * gradient_accumulation_steps)
        #                 = 12600 / (16 * 4) = 196.875
        # total_steps = steps_per_epoch * num_train_epochs = 196.875 * 10 = ~1968
        # eval_steps = total_steps * eval_steps_factor = 1968 * 0.1 = ~196
        # warmup_steps = total_steps * warmup_steps_factor = 1968 * 0.2 = ~393
        # save_steps = eval_steps * save_steps_multiple = 196 * 3 = 588
        ##############################
        training_output_dir = training_output_dir + (
            "_small" if use_small_dataset else ""
        )
        dataset_output_dir = dataset_output_dir + (
            "_small" if use_small_dataset else ""
        )
        params = {
            # Save folder paths
            "model_checkpoint": model_checkpoint,
            "dataset_name": "knkarthick/dialogsum",
            "output_dir": f"./{training_output_dir}/training-output",
            "logging_dir": f"./{training_output_dir}/training-logs",
            "cache_dir": f"./{training_output_dir}/cache",
            "model_save_path": f"./{training_output_dir}/finetuned-output",
            "params_save_path": f"./{training_output_dir}/training_params.json",
            "dataset_save_path": f"./{dataset_output_dir}/dataset-preprocessing-output",
            # Do not change
            "run_name": None,
            # Hyperparameters
            "prefix": "",  # In case of T5 model, use 'summarize: '
            "max_source_length": 512,  # Check dataset-analysis.ipynb for max_source_length
            "max_target_length": 90,  # Check dataset-analysis.ipynb for max_target_length
            # TrainingArguments
            "metric_for_best_model": "eval_loss",
            "fp16": True,
            "weight_decay": 0.06,
            "learning_rate": 0.000001,
            "per_device_train_batch_size": 12,
            "gradient_accumulation_steps": 6,
            "num_train_epochs": 15,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.18,
            "eval_steps": 0.1,
            "save_steps": 0.2,
            "logging_steps": 0.02,
            # GenerationConfig
            "generation_config": (
                {
                    "_from_model_config": False,
                    "max_length": 90,
                    "min_length": 1,
                    "num_beams": 6,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 1.6,
                }
            ),
            # Extra params
            "use_generation_config": False,
            "use_early_stopping": True,
            "early_stopping_patience": 3,
            "use_small_dataset": use_small_dataset,
            "n_trials": n_trials,
            "sweep_id": sweep_id,
            "metrics": ["rouge", "bleu"],
        }

        return params

    # Fine-tuning the model ##############
    # Change this only if you are changing tokenization process, max_source_length, or max_target_length
    dataset_output_folder = "dialogsum_tokenized_dataset"

    # Model to load and fine-tune
    model_checkpoint = "facebook/bart-large-xsum"
    # model_checkpoint = "facebook/bart-large-cnn"
    # model_checkpoint = "facebook/bart-large"
    training_output_dir = "bart_hp_search_output"

    params = get_params(
        model_checkpoint,
        training_output_dir,
        dataset_output_folder,
        use_small_dataset=False,  # If True, then probably use like 2-3 n_trials in params
        n_trials=None,  # If you are resuming a sweep then update the n_trials by subtracting number of completed trials. Ex - n_trials = 10, already run = 6, then update n_trials = 4
        sweep_id=None,  # If you want to resume a sweep, else pass None
    )
    fine_tune_model(params)
