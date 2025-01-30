import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
import math
import json

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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


def compute_pf(emotion_counts, emotion, smoothing_factor=1e-9):
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

    freq_p_e = emotion_counts.get(
        emotion, 0
    )  # Frequency of emotion in the current document
    pf[emotion] = (freq_p_e) / (total_emotion_freq + 1)  # PF formula
    return pf


def compute_ief(documents, emotion, smoothing_factor=1e-9):
    """
    Computes the Inverse Emotion Frequency (IEF) for each emotion across all documents.

    Args:
        documents (list): A list of dictionaries, each representing emotion counts in a document.

    Returns:
        dict: IEF values for each emotion.
    """
    N = len(documents)  # Total number of documents
    freq = 0

    # Count how many documents contain each emotion
    for doc in documents:
        if emotion in doc:
            freq += 1

    print("freq", freq)

    # Compute IEF using the PF-IEF formula
    ief = {}
    ief[emotion] = math.log(1 + N / (freq + 1))
    return ief


def compute_pf_ief(emotion_counts, ief, emotion):
    """
    Computes the PF-IEF weighted emotion vector for a document.

    Args:
        emotion_counts (dict): Emotion frequency counts for the document.
        ief (dict): Precomputed IEF values for each emotion.

    Returns:
        dict: PF-IEF weighted emotion vector.
    """
    pf = compute_pf(emotion_counts, emotion)
    # PF-IEF calculation: Multiply PF by IEF for each emotion
    pf_ief = {emotion: pf.get(emotion, 0) * ief.get(emotion, 0)}
    return pf_ief


def normalize_emotions(emotion_counts):
    """
    Helper function to normalize emotion counts into probability distributions.
    """
    total_count = sum(emotion_counts.values())
    if total_count == 0:
        return {emotion: 0 for emotion in emotion_labels_no_neutral.values()}
    return {emotion: count / total_count for emotion, count in emotion_counts.items()}


def calculate_kl_divergence(p, q, smoothing_factor=1e-9):
    """
    Function to calculate KL divergence with smoothing.
    """
    # Add smoothing factor to avoid zero probability values
    p = {
        key: p.get(key, 0) + smoothing_factor
        for key in emotion_labels_no_neutral.values()
    }
    q = {
        key: q.get(key, 0) + smoothing_factor
        for key in emotion_labels_no_neutral.values()
    }

    # Normalize the distributions
    p_norm = normalize_emotions(p)
    q_norm = normalize_emotions(q)

    return entropy(list(p_norm.values()), list(q_norm.values()))


def calculate_cosine_similarity(tfidf_dialogue, tfidf_summary):
    """
    Calculate cosine similarity between two vectors.
    """
    # print(tfidf_dialogue)
    # print(tfidf_summary)
    # print("--" * 50)
    return cosine_similarity(
        [list(tfidf_dialogue.values())], [list(tfidf_summary.values())]
    )[0][0]


def process_emotions_individual(emotions_list, summary_type="human"):
    """
    Process emotions for each individual emotion label to calculate metrics separately.

    Args:
        emotions_list (list): List of dictionaries containing emotion data for dialogue and summary.
        summary_type (str): The type of summary to compare ("human" or "machine").

    Returns:
        dict: Metrics (cosine similarity and KL divergence) for each individual emotion.
    """
    emotion_metrics = {}

    for emotion in list(emotion_labels.values()):
        # if emotion != "neutral":
        #     continue
        print(f"Processing emotion: {emotion}...")
        cosine_similarities = []
        kl_losses = []

        # Prepare documents for PF-IEF calculation
        corpus_dialogue = []
        corpus_summary = []

        for emotions in emotions_list:
            corpus_dialogue.append(emotions["dialogue_emotions"])
            corpus_summary.append(emotions[f"{summary_type}_summary_emotions"])

        ief_dialogue = compute_ief(corpus_dialogue, emotion)
        ief_summary = compute_ief(corpus_summary, emotion)

        for i in range(len(emotions_list)):
            # if i != 20:
            #     continue
            # print(i)

            dialogue_emotions = corpus_dialogue[i]
            summary_emotions = corpus_summary[i]

            # print("dialogue_emotions", dialogue_emotions)
            # print("summary_emotions", summary_emotions)

            # Compute PF-IEF for each document
            pf_ief_dialogue = compute_pf_ief(dialogue_emotions, ief_dialogue, emotion)
            pf_ief_summary = compute_pf_ief(summary_emotions, ief_summary, emotion)

            # print("pf_ief_dialogue", pf_ief_dialogue)
            # print("pf_ief_summary", pf_ief_summary)

            # Calculate cosine similarity and KL divergence
            cosine_sim = calculate_cosine_similarity(pf_ief_dialogue, pf_ief_summary)
            kl_loss = calculate_kl_divergence(dialogue_emotions, summary_emotions)

            cosine_similarities.append(cosine_sim)
            kl_losses.append(kl_loss)

        # Average the metrics for the current emotion
        mean_cosine_sim = np.mean(cosine_similarities)
        avg_kl_loss = np.mean(kl_losses)

        emotion_metrics[emotion] = {
            "mean_cosine_sim": mean_cosine_sim,
            "avg_kl_loss": avg_kl_loss,
        }

    return emotion_metrics


def calculate_eres_individual(emotions_list_path, summary_type, alpha=0.5, lambda_=1.0):
    """
    Calculate the Emotion Retention Score (ERES) for each individual emotion.

    Args:
        emotions_list_path (str): Path to the JSON file containing emotions data.
        alpha (float): Weighting factor for RACS.
        lambda_ (float): Scaling factor for KL divergence.
        summary_type (str): The type of summary to compare ("human" or "machine").

    Returns:
        dict: ERES, RACS, and AACS for each individual emotion.
    """
    print(f"Loading emotions list from {emotions_list_path}...")
    print(f"Calculating for {summary_type} summaries...")
    with open(emotions_list_path, "r") as file:
        emotions_list = json.load(file)

    # Process emotions for individual metrics
    emotion_metrics = process_emotions_individual(emotions_list, summary_type)

    eres_results = {}
    for emotion, metrics in emotion_metrics.items():
        avg_cs = metrics["mean_cosine_sim"]
        avg_kl = metrics["avg_kl_loss"]

        # Normalize Spearman to get RACS
        RACS = (avg_cs + 1) / 2

        # Calculate AACS
        AACS = np.exp(-lambda_ * avg_kl)

        # Calculate ERES
        ERES = alpha * RACS + (1 - alpha) * AACS

        eres_results[emotion] = {
            "RACS": RACS,
            "AACS": AACS,
            "ERES": ERES,
        }

    return eres_results


# Example usage
results = calculate_eres_individual(
    emotions_list_path="./sentence_level_emotion_results_bart_ds.json",
    summary_type="machine",
)

for emotion, metrics in results.items():
    print(f"Emotion: {emotion}")
    print(f"  RACS: {metrics['RACS']:.3f}")
    print(f"  AACS: {metrics['AACS']:.3f}")
    print(f"  ERES: {metrics['ERES']:.3f}")
