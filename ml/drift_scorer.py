import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_time_decay(weeks, current_week, decay_rate=0.1):
    """
    Give more importance to recent student activities.
    """
    weeks = np.array(weeks, dtype=float)

    time_difference = np.maximum(current_week - weeks, 0)

    weights = np.exp(-decay_rate * time_difference)

    return weights

def calculate_weighted_interest(
    activity_weeks,
    activity_vectors,
    current_week,
    decay_rate=0.1
):
    """
    Calculate student interest using time-decay weights.
    """
    weights = calculate_time_decay(
        activity_weeks,
        current_week,
        decay_rate
    )

    total_weight = np.sum(weights)

    if total_weight == 0:
        return np.zeros(activity_vectors.shape[1])

    weighted_interest = np.average(
        activity_vectors.toarray(),
        axis=0,
        weights=weights
    )
    return weighted_interest

def calculate_preference_drift(
    activity_weeks,
    activity_vectors,
    current_week,
    decay_rate=0.1
):
    """
    Compare recent interests with older interests.
    """
    activity_weeks = np.array(activity_weeks)

    recent_mask = activity_weeks >= current_week - 4
    older_mask = activity_weeks < current_week - 4

    if not np.any(recent_mask) or not np.any(older_mask):
        return 0.0

    recent_interest = calculate_weighted_interest(
        activity_weeks[recent_mask],
        activity_vectors[recent_mask],
        current_week,
        decay_rate
    )

    older_interest = calculate_weighted_interest(
        activity_weeks[older_mask],
        activity_vectors[older_mask],
        current_week - 4,
        decay_rate
    )

    similarity = cosine_similarity(
        recent_interest.reshape(1, -1),
        older_interest.reshape(1, -1)
    )[0][0]

    drift_score = 1 - similarity

    return round(float(np.clip(drift_score, 0, 1)), 4)

def rerank_recommendations(
    candidate_vectors,
    student_interest,
    drift_score
):
    """
    Rank candidate resources using student interest.
    """
    scores = cosine_similarity(
        candidate_vectors,
        student_interest.reshape(1, -1)
    ).flatten()

    # Adjust scores based on detected preference drift.
    adjusted_scores = scores * (1 + drift_score)

    return adjusted_scores

if __name__ == "__main__":
    from ml.data_loader import load_datasets
    from ml.text_processor import prepare_resource_features
    from ml.tfidf_model import build_tfidf_model

    datasets = load_datasets()

    vle = datasets["vle"].reset_index(drop=True)

    processed = prepare_resource_features(vle)

    vectorizer, tfidf_matrix = build_tfidf_model(processed)

    # Demo using resource week information.
    weeks = vle["week_from"].fillna(0).to_numpy()

    current_week = int(np.max(weeks))

    student_interest = calculate_weighted_interest(
        weeks,
        tfidf_matrix,
        current_week
    )

    drift = calculate_preference_drift(
        weeks,
        tfidf_matrix,
        current_week
    )

    recommendations = rerank_recommendations(
        tfidf_matrix,
        student_interest,
        drift
    )

    print("\nTIME DECAY AND DRIFT SCORER COMPLETED")
    print("Current week:", current_week)
    print("Preference drift score:", drift)
    print("Recommendation scores generated:", len(recommendations))