from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)
from collections import Counter
import nltk
from tqdm import tqdm
import json
import os
import torch


# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define emotion mapping based on your BERT model's labels
# ["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"]
emotion_labels = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

# Load DialogSum dataset
dataset = load_dataset("knkarthick/dialogsum")
test_dataset = dataset["test"]
test_dataset = test_dataset.select(range(1, test_dataset.num_rows, 3))

# model_checkpoint = "./finetuned_bart_dialogsum/checkpoint-2925"
model_checkpoint = "facebook/bart-base"

# Load models
bart_model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
bart_model.to(device)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bert_model = BertForSequenceClassification.from_pretrained(
    "./finetuned_bert_carer_goemotions"
)
bert_model.to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Helper functions
def generate_summary(dialogue):
    """Generate machine summary using the BART model."""
    inputs = bart_tokenizer(
        dialogue, return_tensors="pt", max_length=256, truncation=True
    ).to(device)
    summary_ids = bart_model.generate(
        inputs["input_ids"], max_length=128, num_beams=5, early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def extract_emotions(sentences):
    """Extract aggregated emotions from individual sentences."""
    emotion_counts = Counter()
    sentence_emotions = []
    for sentence in sentences:
        # Tokenize sentence for the emotion model
        inputs = bert_tokenizer(
            sentence, return_tensors="pt", padding="max_length", truncation=True
        ).to(device)
        outputs = bert_model(**inputs)
        emotion = outputs.logits.argmax(dim=1).item()  # Predicted emotion index
        emotion_label = emotion_labels[emotion]
        emotion_counts[emotion_label] += 1
        sentence_emotions.append({"sentence": sentence, "emotion": emotion_label})
    return dict(emotion_counts), sentence_emotions


def tokenize_text(text):
    """Tokenize text into sentences."""
    return nltk.sent_tokenize(text)


def split_by_newline(text):
    """Split text into sentences/lines based on newlines."""
    return [line.strip() for line in text.split("\n") if line.strip()]


# Process selected test dialogues
results = []
for sample in tqdm(test_dataset):
    dialogue = sample["dialogue"]
    human_summary = sample["summary"]

    # Tokenize dialogues and summaries into individual sentences
    dialogue_sentences = tokenize_text(dialogue)
    human_summary_sentences = tokenize_text(human_summary)

    # Generate machine summary and tokenize it
    machine_summary = generate_summary(dialogue)
    machine_summary_sentences = tokenize_text(machine_summary)

    # Extract and aggregate emotions from individual sentences
    dialogue_emotions, dialogue_sentence_emotions = extract_emotions(dialogue_sentences)
    human_summary_emotions, human_summary_sentence_emotions = extract_emotions(
        human_summary_sentences
    )
    machine_summary_emotions, machine_summary_sentence_emotions = extract_emotions(
        machine_summary_sentences
    )

    # Append results
    results.append(
        {
            "dialogue_emotions": dialogue_emotions,
            "human_summary_emotions": human_summary_emotions,
            "machine_summary_emotions": machine_summary_emotions,
            "dialogue_sentence_emotions": dialogue_sentence_emotions,
            "human_summary_sentence_emotions": human_summary_sentence_emotions,
            "machine_summary_sentence_emotions": machine_summary_sentence_emotions,
        }
    )

# Save results
output_path = "sentence_level_emotion_results_3.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {output_path}")


# print("Results:")
# print(json.dumps(results, indent=2))
