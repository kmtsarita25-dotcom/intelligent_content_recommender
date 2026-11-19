import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from ml.text_processor import prepare_resource_features
from ml.tfidf_model import build_tfidf_model

def diversify_recommendations(
    candidate_indices,
    scores,
    tfidf_matrix,
    vle,
    top_n=5,
    relevance_weight=0.85,
    type_diversity_weight=0.00
):
    """Select relevant and diverse resources using improved MMR."""

    selected = []
    remaining = list(candidate_indices)

    selected_types = set()

    while remaining and len(selected) < top_n:

        best_index = None
        best_score = -float("inf")

        for index in remaining:

            # Relevance score
            relevance = scores[index]

            # Similarity penalty
            if selected:
                redundancy = cosine_similarity(
                    tfidf_matrix[index],
                    tfidf_matrix[selected]
                ).max()
            else:
                redundancy = 0.0

            # Activity type diversity
            activity_type = vle.iloc[index]["activity_type"]

            if activity_type in selected_types:
                type_bonus = 0.0
            else:
                type_bonus = 1.0

            # Combined MMR score
            mmr_score = (
                relevance_weight * relevance
                - (1 - relevance_weight - type_diversity_weight)
                * redundancy
                + type_diversity_weight * type_bonus
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        if best_index is None:
            break

        selected.append(best_index)

        selected_types.add(
            vle.iloc[best_index]["activity_type"]
        )

        remaining.remove(best_index)

    recommendations = []

    for index in selected:
        recommendations.append({
            "id_site": int(vle.iloc[index]["id_site"]),
            "activity_type": vle.iloc[index]["activity_type"],
            "recommendation_score": round(
                float(scores[index]), 4
            )
        })

    return recommendations

def generate_recommendations(
    student_id,
    datasets,
    top_n=5,
    decay_rate=0.1,
    interest_weight=0.1
):
    """
    Hybrid personalized recommendation system.

    Combines:
    1. Student interest similarity
    2. Course resource popularity
    3. Time-decay weighting
    4. MMR diversity ranking
    """

    if not 0 <= interest_weight <= 1:
        raise ValueError("interest_weight must be between 0 and 1")

    if decay_rate < 0:
        raise ValueError("decay_rate cannot be negative")

    student_vle = datasets["student_vle"].copy()
    vle = datasets["vle"].reset_index(drop=True).copy()

    student_vle["date"] = pd.to_numeric(
        student_vle["date"], errors="coerce"
    )

    student_vle["sum_click"] = pd.to_numeric(
        student_vle["sum_click"], errors="coerce"
    ).fillna(0)

    activity = student_vle[
        student_vle["id_student"] == student_id
    ].copy()

    if activity.empty:
        raise ValueError("No activity found for this student")

    # Find the most active course from available history.
    course = (
        activity.groupby(
            ["code_module", "code_presentation"]
        )["sum_click"]
        .sum()
        .idxmax()
    )

    module, presentation = course

    activity = activity[
        (activity["code_module"] == module) &
        (activity["code_presentation"] == presentation)
    ].copy()

    course_vle = vle[
        (vle["code_module"] == module) &
        (vle["code_presentation"] == presentation)
    ].reset_index(drop=True)

    if course_vle.empty:
        raise ValueError("No resources found for this course")

    print("Preparing resource features...")

    processed = prepare_resource_features(course_vle)

    _, tfidf_matrix = build_tfidf_model(processed)

    # Build student resource history.
    history = (
        activity.groupby("id_site")
        .agg(
            total_clicks=("sum_click", "sum"),
            last_date=("date", "max")
        )
        .reset_index()
    )

    resource_positions = {
        resource_id: index
        for index, resource_id in enumerate(course_vle["id_site"])
    }

    history = history[
        history["id_site"].isin(resource_positions)
    ].copy()

    if history.empty:
        raise ValueError("No matching student resources found")

    history["vector_index"] = history["id_site"].map(
        resource_positions
    )

    # Calculate time-decay weights.
    current_day = history["last_date"].max()

    age_in_weeks = (
        current_day - history["last_date"]
    ) / 7.0

    time_weights = np.exp(-decay_rate * age_in_weeks)

    click_weights = (
        history["total_clicks"].to_numpy() *
        time_weights.to_numpy()
    )

    if click_weights.sum() <= 0:
        raise ValueError("Student activity weights are zero")

    # Build personalized interest profile.
    indices = history["vector_index"].to_numpy()

    student_vectors = tfidf_matrix[indices].toarray()

    student_interest = np.average(
        student_vectors,
        axis=0,
        weights=click_weights
    )

    # Calculate similarity scores.
    interest_scores = cosine_similarity(
        tfidf_matrix,
        student_interest.reshape(1, -1)
    ).flatten()

    # Calculate course popularity up to current day.
    course_activity = student_vle[
        (student_vle["code_module"] == module) &
        (student_vle["code_presentation"] == presentation) &
        (student_vle["date"] <= current_day)
    ]

    popularity = (
        course_activity.groupby("id_site")["sum_click"]
        .sum()
    )

    popularity_scores = (
        course_vle["id_site"]
        .map(popularity)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # Log normalization.
    popularity_scores = np.log1p(popularity_scores)

    max_popularity = popularity_scores.max()

    if max_popularity > 0:
        popularity_scores /= max_popularity

    # Combine interest and popularity.
    hybrid_scores = (
        interest_weight * interest_scores
        + (1 - interest_weight) * popularity_scores
    )

    # Exclude already visited resources.
    visited_ids = set(history["id_site"])

    candidate_indices = [
        index
        for index, resource_id in enumerate(course_vle["id_site"])
        if resource_id not in visited_ids
    ]

    if not candidate_indices:
        return []

    print("Calculating hybrid recommendation scores...")
    print("Applying diversity ranking...")

    recommendations = diversify_recommendations(
        candidate_indices,
        hybrid_scores,
        tfidf_matrix,
        course_vle,
        top_n=top_n
    )

    return recommendations

if __name__ == "__main__":

    from ml.data_loader import load_datasets

    datasets = load_datasets()

    student_id = int(
        datasets["student_vle"]["id_student"].iloc[0]
    )

    print("\nHYBRID PERSONALIZED RECOMMENDATION SYSTEM")
    print("Student ID:", student_id)

    recommendations = generate_recommendations(
        student_id,
        datasets,
        top_n=5
    )

    print("\nRecommended resources:")

    for item in recommendations:
        print(item)

    print("\nTotal recommendations:", len(recommendations))