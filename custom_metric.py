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

emotions_no_neutral = list(
    emotion_labels_no_neutral.values()
)  # Predefined list of all emotions
all_emotions = list(emotion_labels.values())  # Predefined list of all emotions


def compute_ef(emotion_counts):
    """
    Computes the Pattern Frequency (EF) for a document.

    Args:
        emotion_counts (dict): A dictionary with emotions as keys and their frequency in the document as values.

    Returns:
        dict: EF values for each emotion.
    """
    total_emotion_freq = sum(
        emotion_counts.values()
    )  # Total frequency of all emotions in the document
    ef = {}

    for emotion in all_emotions:
        freq_p_e = emotion_counts.get(
            emotion, 0
        )  # Frequency of emotion in the current document
        ef[emotion] = math.log((total_emotion_freq + 1) / (freq_p_e + 1))  # EF formula

    return ef


def compute_ief(documents):
    """
    Computes the Inverse Emotion Frequency (IEF) for each emotion across all documents.

    Args:
        documents (list): A list of dictionaries, each representing emotion counts in a document.

    Returns:
        dict: IEF values for each emotion.
    """
    N = len(documents)  # Total number of documents
    emotion_doc_freq = {emotion: 0 for emotion in all_emotions}

    # Count how many documents contain each emotion
    for doc in documents:
        seen_emotions = [
            e for e in set(doc.keys()) if e in all_emotions
        ]  # Unique emotions in the document

        for emotion in seen_emotions:
            emotion_doc_freq[emotion] += 1

    # Compute IEF using the ef-ief formula
    ief = {}
    for emotion, freq in emotion_doc_freq.items():
        ief[emotion] = math.log(1 + N / (freq + 1))

    return ief


def compute_ef_ief(emotion_counts, ief):
    """
    Computes the EF-IEF weighted emotion vector for a document.

    Args:
        emotion_counts (dict): Emotion frequency counts for the document.
        ief (dict): Precomputed IEF values for each emotion.

    Returns:
        dict: EF-IEF weighted emotion vector.
    """
    ef = compute_ef(emotion_counts)
    # EF-IEF calculation: Multiply EF by IEF for each emotion
    ef_ief = {emotion: ef[emotion] * ief.get(emotion, 0) for emotion in all_emotions}
    return ef_ief


# Helper function to normalize emotion counts into probability distributions
def normalize_emotions(emotion_counts):
    total_count = sum(emotion_counts.values())
    if total_count == 0:
        return {emotion: 0 for emotion in all_emotions}
    return {emotion: count / total_count for emotion, count in emotion_counts.items()}


# Function to calculate KL divergence with smoothing
def calculate_kl_divergence(p, q, smoothing_factor=1e-9):

    # Add smoothing factor to avoid zero probability values
    p = {key: p.get(key, 0) + smoothing_factor for key in all_emotions}
    q = {key: q.get(key, 0) + smoothing_factor for key in all_emotions}

    # Normalize the distributions
    p_norm = normalize_emotions(p)
    q_norm = normalize_emotions(q)

    return entropy(list(p_norm.values()), list(q_norm.values()))


def calculate_cosine_similarity(tfidf_dialogue, tfidf_summary):
    return 1 - cosine(list(tfidf_dialogue.values()), list(tfidf_summary.values()))


# Function to process the list of emotions and calculate Spearman and similarity
def process_emotions(emotions_list, summary_type="human"):
    cosine_similarities_dialogue_summary = []
    kl_losses_dialogue_summary = []

    # Prepare the documents for EF-IEF calculation
    corpus_dialogue = []
    corpus_summary = []

    for emotions in emotions_list:
        corpus_dialogue.append(emotions["dialogue_emotions"])
        corpus_summary.append(emotions[f"{summary_type}_summary_emotions"])

    ief_dialogue = compute_ief(corpus_dialogue)
    ief_summary = compute_ief(corpus_summary)

    # Calculate the cosine similarities and EF-IEF for each document
    for i in range(len(emotions_list)):
        dialogue_emotions = corpus_dialogue[i]
        summary_emotions = corpus_summary[i]

        # Compute EF-IEF for each document
        ef_ief_dialogue = compute_ef_ief(dialogue_emotions, ief_dialogue)
        ef_ief_summary = compute_ef_ief(summary_emotions, ief_summary)

        # Calculate cosine similarity between Dialogue and Human, and Dialogue and Machine
        cosine_sim_dialogue_summary = calculate_cosine_similarity(
            ef_ief_dialogue, ef_ief_summary
        )

        # Calculate KL divergence (KL Loss) between Dialogue and Human, and Dialogue and Machine
        kl_loss_dialogue_summary = calculate_kl_divergence(
            dialogue_emotions, summary_emotions
        )

        cosine_similarities_dialogue_summary.append(cosine_sim_dialogue_summary)
        kl_losses_dialogue_summary.append(kl_loss_dialogue_summary)

    # Average KL loss across all records
    mean_cosine_sim = np.mean(cosine_similarities_dialogue_summary)
    avg_kl_loss = np.mean(kl_losses_dialogue_summary)

    return mean_cosine_sim, avg_kl_loss


def calculate_eres(emotions_list_path, alpha=0.5, lambda_=1.0, summary_type="human"):
    """
    Calculate the Emotion Retention Score (ERES) using the formula:
    ERES = 𝛼 · RACS + (1 − 𝛼) · exp(−𝜆 · AACS)

    Emotions list should be a JSON file with the following structure:
    [
        {
            "dialogue_emotions": {
                "neutral": 24,
                "fear": 1
            },
            "human_summary_emotions": {
                "neutral": 1
            },
            "machine_summary_emotions": {
                "neutral": 2
            },
            ...
    ]

    :param alpha: The parameter controlling the relative importance of RACS (default is 0.5).
    :param lambda_: The scaling factor for KL divergence normalization (default is 1.0).

    :return: ERES: The final Emotion Retention Score (),
             RACS: The normalized Relative Affective Content Similarity (Spearman correlation value),
             AACS: The Absolute Affective Content Similarity (KL divergence value),
    """

    print(f"Loading emotions list from {emotions_list_path}...")
    with open(emotions_list_path, "r") as file:
        emotions_list = json.load(file)

    # Print the results
    print(f"Summary Type: {summary_type.capitalize()}")

    # Process the emotions list and calculate Spearman correlation and KL loss
    avg_cs, avg_kl = process_emotions(emotions_list, summary_type)

    # Normalize spearman to get RACS
    RACS = (avg_cs + 1) / 2

    AACS = np.exp(-lambda_ * avg_kl)

    ERES = alpha * RACS + (1 - alpha) * AACS

    return RACS, AACS, ERES


# ERES_human = calculate_eres(RACS, AACS_human)
RACS, AACS, ERES = calculate_eres(
    # emotions_list_path="./results/sentence_level_emotion_results_bart_large_ds.json",
    # emotions_list_path="./results/sentence_level_emotion_results_bart_large_cnn.json",
    emotions_list_path="./results/sentence_level_emotion_results_t5_base.json",
    summary_type="machine",
    lambda_=1.0,
)
print(f"RACS: {RACS:.3f}")
# print(f"Average KL Loss (Human Summary): {AACS_human:.3f}")
print(f"AACS: {AACS:.3f}")
# print(f"ERES (Human Summary): {ERES_human:.3f}")
print(f"ERES: {ERES:.3f}")
