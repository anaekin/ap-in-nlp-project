import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset, concatenate_datasets, ClassLabel, DatasetDict
from evaluate import load
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
import os
import gc
import json
import re
import emoji  # Library to handle emojis

# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

wandb.init(mode="disabled")

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def sample_neutral_examples(dataset, min_length=10, max_length=300):
    # Filter based on text length within the range of mean ± num_std_dev * std_dev
    filtered_samples = dataset.filter(
        lambda x: min_length <= len(x["text"]) <= max_length
    )

    # Randomly sample to get between n_min and n_max examples
    neutral_train = filtered_samples["train"].shuffle(seed=42).select(range(2000))
    neutral_val = filtered_samples["validation"].shuffle(seed=42).select(range(250))
    neutral_test = filtered_samples["test"].shuffle(seed=42).select(range(250))

    return DatasetDict(
        {
            "train": neutral_train,
            "validation": neutral_val,
            "test": neutral_test,
        }
    )


def clean_text(text, remove_emojis=True):
    """
    Clean the text by performing basic preprocessing steps:
    - Convert to lowercase
    - Remove special characters and unwanted tokens like URLs, mentions, hashtags
    - Remove emojis (optional)
    - Remove extra spaces
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs (http://, https://, www)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Remove mentions (e.g., @username)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (optional, if irrelevant)
    text = re.sub(r"#\w+", "", text)

    # Remove non-alphanumeric characters (optional, but useful for cleaning)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Optionally remove emojis (if remove_emojis=True)
    if remove_emojis:
        text = emoji.replace_emoji(text, replace="")  # Removes emojis
    else:
        text = emoji.demojize(
            text
        )  # Converts emojis to text (e.g., 😀 -> :grinning_face:)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_go_emotions(examples):
    examples["text"] = [clean_text(text) for text in examples["text"]]
    return examples


def load_datasets():
    carer = load_dataset("dair-ai/emotion")
    goemotions = load_dataset("google-research-datasets/go_emotions")

    # Filter "neutral" samples from GoEmotions and map to label 6
    neutral_samples = goemotions.filter(lambda x: 27 in x["labels"])
    neutral_samples = sample_neutral_examples(neutral_samples)
    neutral_samples = neutral_samples.map(
        lambda x: {"label": 6, "text": x["text"]},
        remove_columns=["labels", "id"],
    )

    updated_class_labels = ClassLabel(
        names=["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]
    )
    carer = carer.cast_column(
        "label",
        updated_class_labels,
    )
    neutral_samples = neutral_samples.cast_column(
        "label",
        updated_class_labels,
    )
    neutral_samples = neutral_samples.map(preprocess_go_emotions, batched=True)

    # Combine CARER and neutral samples
    train_dataset = concatenate_datasets(
        [carer["train"], neutral_samples["train"]]
    ).shuffle(seed=42)
    val_dataset = concatenate_datasets(
        [carer["validation"], neutral_samples["validation"]]
    ).shuffle(seed=42)
    test_dataset = concatenate_datasets(
        [carer["test"], neutral_samples["test"]]
    ).shuffle(seed=42)

    return train_dataset, val_dataset, test_dataset


# Tokenization function
def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)


# Function to preprocess datasets
def preprocess_datasets(tokenizer, train_dataset, val_dataset, test_dataset):
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    # Remove text column and rename label to labels
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset = dataset.remove_columns(["text"])
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch")

    return train_dataset, val_dataset, test_dataset


def clear_cache(model=None, tokenizer=None, trainer=None):
    """
    Clear CUDA memory and delete model, tokenizer, and trainer if they exist.
    """
    print("Freeing CUDA memory...")
    if model:
        del model
    if tokenizer:
        del tokenizer
    if trainer:
        del trainer
    torch.cuda.empty_cache()
    gc.collect()
    print("CUDA cache cleared and GPU memory released.\n")


# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    pred_classes = np.argmax(predictions, axis=1)

    accuracy = load("accuracy").compute(predictions=pred_classes, references=labels)[
        "accuracy"
    ]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_classes, average="weighted"
    )
    try:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError:
        auroc = float("nan")

    report = classification_report(
        labels, pred_classes, output_dict=True, zero_division=0
    )
    per_class_f1 = {
        f"class_{cls}_f1": report[str(cls)]["f1-score"] for cls in range(probs.shape[1])
    }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        **per_class_f1,
    }


# Run the training pipeline
if __name__ == "__main__":
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("./finetuned_bert_carer_goemotions")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./finetuned_bert_carer_goemotions", num_labels=7
    ).to(device)

    # Preprocess datasets
    train_dataset, val_dataset, test_dataset = preprocess_datasets(
        tokenizer, train_dataset, val_dataset, test_dataset
    )

    # Testing Phase
    print("Testing the fine-tuned model on the test set...")

    # Reload model for testing
    # Evaluate on the test dataset
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    preds, labels, test_results = trainer.predict(test_dataset)
    print("Test Results:")
    print(json.dumps(test_results, indent=4))

    # Clear GPU memory
    clear_cache(model, tokenizer, trainer)
    print("Model fine-tuned, tested, and GPU memory cleared successfully!")
