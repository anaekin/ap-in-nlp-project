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


# Step 2: Compute Inverse Emotion Frequency (IEF) across the entire dataset
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
        seen_emotions = set(doc.keys())  # Unique emotions in the document
        for emotion in seen_emotions:
            emotion_doc_freq[emotion] += 1

    # Compute IEF using the pf-ief formula
    ief = {}
    for emotion, freq in emotion_doc_freq.items():
        ief[emotion] = math.log((freq + 1) / (N + 1))  # IEF formula

    return ief


# Step 3: Compute PF-IEF Score for a Document
def compute_pf_ief(emotion_counts, ief):
    """
    Computes the PF-IEF weighted emotion vector for a document.

    Args:
        emotion_counts (dict): Emotion frequency counts for the document.
        ief (dict): Precomputed IEF values for each emotion.

    Returns:
        dict: PF-IEF weighted emotion vector.
    """
    pf = compute_pf(emotion_counts)

    # PF-IEF calculation: Multiply PF by IEF for each emotion
    pf_ief = {emotion: pf[emotion] * ief.get(emotion, 0) for emotion in all_emotions}

    return pf_ief


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
    ief_dialogue = compute_ief(corpus_dialogue)
    ief_human = compute_ief(corpus_human_summary)
    ief_machine = compute_ief(corpus_machine_summary)

    # Calculate the cosine similarities and TF-IDF for each document
    for i in range(len(emotions_list)):
        dialogue_emotions = corpus_dialogue[i]
        human_summary_emotions = corpus_human_summary[i]
        machine_summary_emotions = corpus_machine_summary[i]

        # Compute TF-IDF for each document
        pf_ief_dialogue = compute_pf_ief(dialogue_emotions, ief_dialogue)
        pf_ief_human = compute_pf_ief(human_summary_emotions, ief_human)
        pf_ief_machine = compute_pf_ief(machine_summary_emotions, ief_machine)

        # Calculate cosine similarity between Dialogue and Human, and Dialogue and Machine
        cosine_sim_dialogue_human = calculate_cosine_similarity(
            pf_ief_dialogue, pf_ief_human
        )
        cosine_sim_dialogue_machine = calculate_cosine_similarity(
            pf_ief_dialogue, pf_ief_machine
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


def calculate_eres(emotions_list_path, alpha=0.5, lambda_=1.0):
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

    # Process the emotions list and calculate Spearman correlation and KL loss
    spearman, kl_div_human, kl_div_machine = process_emotions(emotions_list)

    # Normalize spearman to get RACS
    RACS = (spearman + 1) / 2

    AACS_reference = np.exp(-lambda_ * kl_div_human)
    AACS_generated = np.exp(-lambda_ * kl_div_machine)

    # Apply the ERES formula
    ERES_reference = alpha * RACS + (1 - alpha) * AACS_reference
    ERES_generated = alpha * RACS + (1 - alpha) * AACS_generated

    return RACS, AACS_generated, ERES_generated


# ERES_human = calculate_eres(RACS, AACS_human)
RACS, AACS, ERES = calculate_eres(
    emotions_list_path="./sentence_level_emotion_results_bart_large.json"
)

# Print the results
print(f"RACS: {RACS:.3f}")
# print(f"Average KL Loss (Human Summary): {AACS_human:.3f}")
print(f"AACS: {AACS:.3f}")
# print(f"ERES (Human Summary): {ERES_human:.3f}")
print(f"ERES: {ERES:.3f}")
