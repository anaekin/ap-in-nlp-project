import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from bert_score import score
import evaluate
import os
import numpy as np


def check_device():
    """Check and return the available device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_and_preprocess_data(tokenizer, saved_dataset_path="./tokenized_dialogsum"):

    tokenized_dataset = None

    if os.path.exists(saved_dataset_path):
        print(f"Loading tokenized DialogSum dataset from {saved_dataset_path}")

        tokenized_dataset = load_from_disk(saved_dataset_path)

        print(f"Tokenized DialogSum dataset loaded from {saved_dataset_path}")
    else:
        print("Directory does not exist. Preprocessing the dataset...")

        """Load and preprocess the DialogSum dataset."""
        dataset = load_dataset("knkarthick/dialogsum")

        # Preprocessing function
        def preprocess_function(batch):
            # Tokenize inputs
            inputs = tokenizer(
                batch["dialogue"], max_length=512, truncation=True, padding="max_length"
            )

            # Tokenize labels
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    batch["summary"],
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                )

            # Convert labels to a single numpy array
            inputs["labels"] = torch.from_numpy(
                np.array(labels["input_ids"], dtype=np.int64)
            )
            return inputs

        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["dialogue", "summary", "id"],
        )

        # Save the tokenized dataset
        tokenized_dataset.save_to_disk(saved_dataset_path)

        print(f"Tokenized DialogSum dataset saved  to {saved_dataset_path}")

    # Set format for PyTorch
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Split the dataset
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def prepare_training_args(output_dir="./bart-dialogsum"):
    """Prepare and return the training arguments."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Evaluate once per epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,  # Batch size per GPU
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=1000,  # Less frequent logging
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=1,  # Set to 2 if batch size 4 is used
        report_to="none",  # Disable external reporting
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    return training_args


def prepare_data_collator(tokenizer, model):
    """Prepare the data collator for sequence-to-sequence tasks."""
    return DataCollatorForSeq2Seq(tokenizer, model=model)


def create_trainer(model, training_args, train_dataset, val_dataset, data_collator):
    """Create the Trainer object."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Updated parameter to handle tokenization
    )
    return trainer


def generate_predictions(trainer, test_dataset):
    """Generate predictions from the trained model."""
    predictions = trainer.predict(test_dataset)
    return predictions


def decode_predictions(predictions, tokenizer):
    """Decode predictions into text format."""
    decoded_preds = tokenizer.batch_decode(
        predictions.predictions, skip_special_tokens=True
    )
    return decoded_preds


def evaluate_model(decoded_preds, test_dataset, model_name="facebook/bart-large-cnn"):
    """Evaluate the model using ROUGE and BERTScore."""
    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    # Evaluate predictions
    references = test_dataset["summary"]
    rouge_results = rouge.compute(predictions=decoded_preds, references=references)

    print("ROUGE-1:", rouge_results["rouge1"].mid.fmeasure)
    print("ROUGE-2:", rouge_results["rouge2"].mid.fmeasure)
    print("ROUGE-L:", rouge_results["rougeL"].mid.fmeasure)

    # Evaluate predictions using BERTScore
    P, R, F1 = score(decoded_preds, references, lang="en", verbose=True)
    print("BERTScore F1:", F1.mean().item())


def save_model(model, tokenizer):
    """Save the fine-tuned model and tokenizer."""
    model.save_pretrained("./bart-dialogsum-finetuned")
    tokenizer.save_pretrained("./bart-dialogsum-finetuned")


def main():
    # Check for GPU
    device = check_device()

    # Initialize tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(tokenizer)

    # Initialize model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model.to(device)  # Move model to GPU

    # Prepare training arguments and data collator
    training_args = prepare_training_args()
    data_collator = prepare_data_collator(tokenizer, model)

    # Create Trainer
    trainer = create_trainer(
        model, training_args, train_dataset, val_dataset, data_collator
    )

    # Train the model
    print(f"Model is on: {model.device}")
    trainer.train()

    # Evaluate the model on the test set
    results = trainer.evaluate(test_dataset)
    print(f"Evaluation Results: {results}")

    # Generate and decode predictions
    predictions = generate_predictions(trainer, test_dataset)
    decoded_preds = decode_predictions(predictions, tokenizer)

    # Print some predictions
    for i in range(5):
        print(f"Dialogue: {test_dataset[i]['dialogue']}")
        print(f"Generated Summary: {decoded_preds[i]}")
        print(f"Reference Summary: {test_dataset[i]['summary']}")
        print()

    # Evaluate the model
    evaluate_model(decoded_preds, test_dataset)

    # Save the fine-tuned model
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
