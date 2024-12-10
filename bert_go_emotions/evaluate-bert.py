import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    roc_curve,
    RocCurveDisplay
)
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define emotion labels
class_names = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")
model.to(device)

# Load the test dataset
test_data = torch.load("test_dataset.pt")

# Convert test data to Hugging Face Dataset format
def convert_to_dataset(data_dict):
    if not all(k in data_dict for k in ["input_ids", "attention_mask", "labels"]):
        raise ValueError("The dictionary doesn't have the expected keys.")
    dataset = Dataset.from_dict({
        "input_ids": torch.tensor(data_dict["input_ids"]),
        "attention_mask": torch.tensor(data_dict["attention_mask"]),
        "labels": torch.tensor(data_dict["labels"]),
    })
    return dataset

test_dataset = convert_to_dataset(test_data)

# Create a Trainer object for evaluation
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
)

# Evaluate on the test set
print("Evaluating the model on the test set...")
test_results = trainer.predict(test_dataset)

# Extract predictions and true labels
y_pred = np.argmax(test_results.predictions, axis=1)  # Convert logits to class indices
y_true = np.array(test_dataset["labels"])  # Convert labels to a NumPy array

# Ensure they are 1D arrays
if y_true.ndim > 1:
    y_true = np.argmax(y_true, axis=1)
if y_pred.ndim > 1:
    y_pred = np.argmax(y_pred, axis=1)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

# Compute AUROC (One-vs-Rest)
try:
    probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1).numpy()
    auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
except ValueError:
    auroc = float('nan')  # Handle cases where AUROC cannot be computed

# Compute per-class metrics
report = classification_report(y_true, y_pred, target_names=class_names.values(), zero_division=0)
print("\nDetailed Classification Report:")
print(report)

# Print overall metrics
print("\nTest set results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")

# Plot AUROC curve
def plot_auroc_curve(y_true, y_prob, class_names):
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names.values()):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f"Class {class_name}").plot(ax=plt.gca())
    
    plt.title("AUROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

plot_auroc_curve(y_true, probs, class_names)

# Example inference
custom_text = "I'm not feeling good today."
inputs = tokenizer(custom_text, return_tensors="pt", truncation=True, padding=True).to(device)
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()
predicted_emotion = class_names[predicted_class]

print(f"\nPredicted emotion for input '{custom_text}': {predicted_emotion}")
