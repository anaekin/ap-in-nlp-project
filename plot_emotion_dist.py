import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import math
import json

# Emotion labels map
emotion_labels = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}
# Exclude neutral emotion from the list
emotion_labels_no_neutral = emotion_labels.copy()
del emotion_labels_no_neutral[6]


all_emotions = list(emotion_labels.values())  # Predefined list of all emotions


# Step 1: Compute Pattern Frequency (PF) for a given document
def compute_pf(emotion_counts):
    """
    Computes the Pattern Frequency (PF) for a document.

    Args:
        emotion_counts (dict): A dictionary with emotions as keys and their frequency in the document as values.

    Returns:
        dict: PF values for each emotion.
    """
    total_emotion_freq = sum(
        emotion_counts.values()
    )  # Total frequency of all emotions in the document
    pf = {}

    for emotion in all_emotions:
        freq_p_e = emotion_counts.get(
            emotion, 0
        )  # Frequency of emotion in the current document
        pf[emotion] = math.log((total_emotion_freq + 1) / (freq_p_e + 1))  # PF formula

    return pf


def plot_bar_graph(data1, data2, data3):
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data for plotting
    categories = list(data1.keys())
    values1 = list(data1.values())
    values2 = list(data2.values())
    values3 = list(data3.values())

    x = np.arange(len(categories))  # the label locations
    width = 0.20  # the width of the bars

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, values1, width, label="Dialogues", color="skyblue")
    bars2 = ax.bar(x, values2, width, label="Human summaries", color="salmon")
    bars3 = ax.bar(
        x + width, values3, width, label="Machine summaries", color="lightgreen"
    )

    # Logarithmic scale
    ax.set_yscale("log")

    # Add labels, title, and legend
    ax.set_xlabel("Emotions", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Emotion Frequencies Across Dialogue and Summaries (Log scale)", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend()

    # Display plot
    plt.tight_layout()
    plt.show()


def compute_overall_distribution(documents):
    overall_distribution = {emotion: 0 for emotion in all_emotions}
    for doc in documents:
        for emotion, count in doc.items():
            overall_distribution[emotion] += count

    return overall_distribution


# def compute_overall_distribution(documents):

#     def get_emotions(emotion_counts):
#         for emotion in all_emotions:
#             freq_p_e = emotion_counts.get(emotion, 0)
#         return freq_p_e

#     overall_distribution = [get_emotions(row) for row in documents]
#     return overall_distribution


# Helper function to normalize emotion counts into probability distributions
def normalize_emotions(emotion_counts):
    total_count = sum(emotion_counts.values())
    if total_count == 0:
        return {emotion: 0 for emotion in emotion_labels_no_neutral.values()}
    return {emotion: count / total_count for emotion, count in emotion_counts.items()}


# Function to calculate KL divergence with smoothing
def calculate_kl_divergence(p, q, smoothing_factor=1e-9):

    # Add smoothing factor to avoid zero probability values
    p = {
        key: p.get(key, 0) + smoothing_factor
        for key in emotion_labels_no_neutral.values()
    }
    q = {
        key: q.get(key, 0) + smoothing_factor
        for key in emotion_labels_no_neutral.values()
    }

    print(p)
    print(q)

    # Normalize the distributions
    p_norm = normalize_emotions(p)
    q_norm = normalize_emotions(q)

    return entropy(list(p_norm.values()), list(q_norm.values()))


# Function to process the list of emotions and calculate Spearman and similarity
def process_emotions(emotions_list):
    # Prepare the documents for TF-IDF calculation
    corpus_dialogue = []
    corpus_human_summary = []
    corpus_machine_summary = []

    for emotions in emotions_list:
        corpus_dialogue.append(emotions["dialogue_emotions"])
        corpus_human_summary.append(emotions["human_summary_emotions"])
        corpus_machine_summary.append(emotions["machine_summary_emotions"])

    print(corpus_dialogue[0])

    dialogue_emo_dist = compute_overall_distribution(corpus_dialogue)
    human_emo_dist = compute_overall_distribution(corpus_human_summary)
    machine_emo_dist = compute_overall_distribution(corpus_machine_summary)

    print(dialogue_emo_dist)
    print(human_emo_dist)
    print(machine_emo_dist)

    plot_bar_graph(dialogue_emo_dist, human_emo_dist, machine_emo_dist)

    # Calculate KL divergence (KL Loss) between Dialogue and Human, and Dialogue and Machine
    kl_loss_dialogue_human = calculate_kl_divergence(dialogue_emo_dist, human_emo_dist)
    kl_loss_dialogue_machine = calculate_kl_divergence(
        dialogue_emo_dist, machine_emo_dist
    )

    print(
        "KL Human",
        (kl_loss_dialogue_human),
        "KL Machine",
        (kl_loss_dialogue_machine),
    )

    print(
        "KL Human",
        math.exp(-(2.0 * kl_loss_dialogue_human)),
        "KL Machine",
        math.exp(-(2.0 * kl_loss_dialogue_machine)),
    )


def main(emotions_list_path):
    print(f"Loading emotions list from {emotions_list_path}...")
    with open(emotions_list_path, "r") as file:
        emotions_list = json.load(file)

    process_emotions(emotions_list)


main(emotions_list_path="./sentence_level_emotion_results_bart_large_ds.json")
# main(emotions_list_path="./sentence_level_emotion_results_bart_large_cnn.json")
# main(emotions_list_path="./sentence_level_emotion_results_t5_base.json")
