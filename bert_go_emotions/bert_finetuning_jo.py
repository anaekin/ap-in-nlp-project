# pip install torch transformers dataset evaluate nltk accelerate absl-py rouge_score bert_score wandb scikit-learn
import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from evaluate import load
import os
import numpy as np
import gc
import nltk
import re
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Initialize WandB
wandb.init(
    project="BERT",
    name="BERT_finetuning",
    config={
        "batch_size": 8,
        "learning_rate": 5e-5,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
    },
)

# Load the accuracy metric
metric = load("accuracy")


def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# Load .pt files into Python dictionary format
def load_pt_data(file_path):
    return torch.load(file_path)


# Convert dictionary data to Dataset format
def convert_to_dataset(pt_data):
    # Extract 'input_ids', 'attention_mask', 'labels' directly from the dictionary
    input_ids = pt_data["input_ids"]
    attention_mask = pt_data["attention_mask"]
    labels = pt_data["labels"]

    # Create a Hugging Face dataset from the dictionary
    dataset = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )
    return dataset


# Enhanced compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)

    # Compute accuracy
    accuracy = metric.compute(predictions=pred_classes, references=labels)["accuracy"]

    # Compute precision, recall, F1 (weighted to handle class imbalance)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average="weighted"
    )

    # Compute AUROC (One-vs-Rest)
    try:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auroc = float("nan")  # Handle cases where AUROC cannot be computed

    # Compute per-class metrics (Optional)
    report = classification_report(
        labels, pred_classes, output_dict=True, zero_division=0
    )
    per_class_f1 = {
        f"class_{cls}_f1": report[str(cls)]["f1-score"] for cls in range(probs.shape[1])
    }

    # Combine metrics into a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        **per_class_f1,  # Add per-class F1 scores
    }

    return metrics


# Function to dynamically calculate steps
def add_steps_params(training_args, dataset_size):
    steps_per_epoch = dataset_size // (
        training_args.per_device_train_batch_size * torch.cuda.device_count()
    )
    training_args.eval_steps = max(1, steps_per_epoch // 2)
    training_args.save_steps = training_args.eval_steps
    training_args.logging_steps = max(1, steps_per_epoch // 4)
    return training_args


# Function to clear the cache and free up GPU memory
def clear_cache(model, tokenizer, trainer):
    print("Freeing CUDA memory...")
    del model
    del tokenizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    print("CUDA cache cleared and GPU memory released.\n")


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=28
)

# Check and set the device
device = check_device()
model.to(device)

# Load the .pt files (assuming they are dictionaries with keys 'input_ids', 'attention_mask', 'labels')
train_data = load_pt_data("train_dataset.pt")
val_data = load_pt_data("val_dataset.pt")
test_data = load_pt_data("test_dataset.pt")

# Convert to Dataset format
train_dataset = convert_to_dataset(train_data)
val_dataset = convert_to_dataset(val_data)
test_dataset = convert_to_dataset(test_data)

# Sample subsets
train_subset = train_dataset.select(range(100))  # First 100 examples
val_subset = val_dataset.select(range(50))  # First 50 examples

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    run_name="bert-finetune-emotions",  # A human-readable name for the wandb run
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_dir="./logs",
    report_to="wandb",  # Enable WandB logging
    per_device_train_batch_size=8,  # Updated batch size
    per_device_eval_batch_size=8,  # Updated eval batch size
    gradient_accumulation_steps=4,  # Updated gradient accumulation steps
    learning_rate=5e-5,  # Updated learning rate
    num_train_epochs=3,
    weight_decay=0.01,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=10,  # Log more frequently for visibility
    save_steps=20,
)

# Adjust dynamic parameters based on dataset size
training_args = add_steps_params(training_args, len(train_dataset))

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Clear GPU memory after training
clear_cache(model, tokenizer, trainer)

print("Model fine-tuned and saved successfully!")

# Finish the WandB run
wandb.finish()
