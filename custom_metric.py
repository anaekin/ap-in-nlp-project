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


# Helper function to compute term frequency (TF) for a given document (emotion counts)
def compute_tf(emotion_counts):
    total_terms = sum(emotion_counts.values())  # total number of emotions
    tf = {
        emotion: emotion_counts.get(emotion, 0) / total_terms if total_terms > 0 else 0
        for emotion in all_emotions
    }
    return tf


# Helper function to compute inverse document frequency (IDF) for each emotion
def compute_idf(documents, debug=False):
    N = len(documents)  # Total number of documents
    idf = {emotion: 0 for emotion in all_emotions}  # Initialize IDF values

    # Count how many documents contain each emotion
    doc_freq = {emotion: 0 for emotion in all_emotions}
    for doc in documents:
        seen_emotions = set(
            doc.keys()
        )  # Avoid counting the same emotion multiple times
        for emotion in seen_emotions:
            if (
                emotion in all_emotions
            ):  # Only count emotions present in the predefined list
                doc_freq[emotion] += 1

    # Calculate IDF for each emotion
    for emotion, freq in doc_freq.items():
        idf[emotion] = math.log(
            (N / (freq + 1)) + 1
        )  # Adding 1 to avoid division by zero for unseen terms

    return idf


# Compute TF-IDF for a single document (dialogue or summary)
def calculate_tfidf(tf, idf):
    tfidf = {
        emotion: tf.get(emotion, 0) * idf.get(emotion, 0) for emotion in all_emotions
    }
    return tfidf


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

    # Normalize the distributions
    p_norm = normalize_emotions(p)
    q_norm = normalize_emotions(q)

    return entropy(list(p_norm.values()), list(q_norm.values()))


def calculate_cosine_similarity(tfidf_dialogue, tfidf_summary):
    return 1 - cosine(list(tfidf_dialogue.values()), list(tfidf_summary.values()))


# Function to process the list of emotions and calculate Spearman and similarity
def process_emotions(emotions_list):
    cosine_similarities_dialogue_human = []
    cosine_similarities_dialogue_machine = []
    kl_losses_dialogue_human = []
    kl_losses_dialogue_machine = []

    # Prepare the documents for TF-IDF calculation
    corpus_dialogue = []
    corpus_human_summary = []
    corpus_machine_summary = []

    for emotions in emotions_list:
        corpus_dialogue.append(emotions["dialogue_emotions"])
        corpus_human_summary.append(emotions["human_summary_emotions"])
        corpus_machine_summary.append(emotions["machine_summary_emotions"])

    # Calculate IDF for the entire corpus
    idf_dialogue = compute_idf(corpus_dialogue)
    idf_human = compute_idf(corpus_human_summary)
    idf_machine = compute_idf(corpus_machine_summary, True)

    # Calculate the cosine similarities and TF-IDF for each document
    for i in range(len(emotions_list)):
        dialogue_emotions = corpus_dialogue[i]
        human_summary_emotions = corpus_human_summary[i]
        machine_summary_emotions = corpus_machine_summary[i]

        # Compute TF for each document
        tf_dialogue = compute_tf(dialogue_emotions)
        tf_human = compute_tf(human_summary_emotions)
        tf_machine = compute_tf(machine_summary_emotions)

        # Compute TF-IDF for each document
        tfidf_dialogue = calculate_tfidf(tf_dialogue, idf_dialogue)
        tfidf_human = calculate_tfidf(tf_human, idf_human)
        tfidf_machine = calculate_tfidf(tf_machine, idf_machine)

        # Calculate cosine similarity between Dialogue and Human, and Dialogue and Machine
        cosine_sim_dialogue_human = calculate_cosine_similarity(
            tfidf_dialogue, tfidf_human
        )
        cosine_sim_dialogue_machine = calculate_cosine_similarity(
            tfidf_dialogue, tfidf_machine
        )

        # Calculate KL divergence (KL Loss) between Dialogue and Human, and Dialogue and Machine
        kl_loss_dialogue_human = calculate_kl_divergence(
            dialogue_emotions, human_summary_emotions
        )
        kl_loss_dialogue_machine = calculate_kl_divergence(
            dialogue_emotions, machine_summary_emotions
        )

        cosine_similarities_dialogue_human.append(cosine_sim_dialogue_human)
        cosine_similarities_dialogue_machine.append(cosine_sim_dialogue_machine)
        kl_losses_dialogue_human.append(kl_loss_dialogue_human)
        kl_losses_dialogue_machine.append(kl_loss_dialogue_machine)

    # Calculate Spearman correlation between dialogue vs human and dialogue vs machine cosine similarities
    spearman_corr = spearmanr(
        cosine_similarities_dialogue_human, cosine_similarities_dialogue_machine
    )[0]

    # Average KL loss across all records
    avg_kl_loss_human = np.mean(kl_losses_dialogue_human)
    avg_kl_loss_machine = np.mean(kl_losses_dialogue_machine)

    return spearman_corr, avg_kl_loss_human, avg_kl_loss_machine


with open("./sentence_level_emotion_results_3.json", "r") as file:
    emotions_list = json.load(file)

# Process the emotions list and calculate Spearman correlation and KL loss
spearman, kl_div_human, kl_div_machine = process_emotions(emotions_list)

# Normalize spearman to get RACS
RACS = (spearman + 1) / 2

lambda_ = 1.0
# AACS_human = np.exp(-lambda_ * kl_div_human)
AACS_machine = np.exp(-lambda_ * kl_div_machine)


def calculate_eres(RACS, AACS, alpha=0.5):
    """
    Calculate the Emotion Retention Score (ERES) using the formula:
    ERES = 𝛼 · RACSnorm + (1 − 𝛼) · exp(−𝜆 · AACS)

    :param RACS: The normalized Relative Affective Content Similarity (Spearman correlation value).
    :param AACS: The Absolute Affective Content Similarity (KL divergence value).
    :param alpha: The parameter controlling the relative importance of RACS (default is 0.5).
    :param lambda_: The scaling factor for KL divergence normalization (default is 1.0).

    :return: The final Emotion Retention Score (ERES).
    """

    # Apply the ERES formula
    eres = alpha * RACS + (1 - alpha) * AACS

    return eres


# ERES_human = calculate_eres(RACS, AACS_human)
ERES_machine = calculate_eres(RACS, AACS_machine)

# Print the results
print(f"RACS: {RACS:.3f}")
# print(f"Average KL Loss (Human Summary): {AACS_human:.3f}")
print(f"AACS: {AACS_machine:.3f}")
# print(f"ERES (Human Summary): {ERES_human:.3f}")
print(f"ERES: {ERES_machine:.3f}")
