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
    relevance_weight=0.75,
    type_diversity_weight=0.10,
):
    """Select relevant and diverse resources using Maximal Marginal Relevance (MMR)."""
    selected = []
    remaining = list(candidate_indices)
    selected_types = set()

    # Ensure weights stay balanced and non-negative
    redundancy_weight = max(0.0, 1.0 - relevance_weight - type_diversity_weight)

    while remaining and len(selected) < top_n:
        best_index = None
        best_score = -float("inf")

        for index in remaining:
            # 1. Base hybrid relevance score
            relevance = float(scores[index])

            # 2. Content redundancy penalty against already-selected items
            if selected:
                redundancy = float(
                    cosine_similarity(
                        tfidf_matrix[index],
                        tfidf_matrix[selected],
                    ).max()
                )
            else:
                redundancy = 0.0

            # 3. Activity-type diversity bonus (e.g., balancing quiz, resource, oucontent)
            activity_type = vle.iloc[index]["activity_type"]
            type_bonus = 0.0 if activity_type in selected_types else 1.0

            # 4. Combined MMR score
            mmr_score = (
                relevance_weight * relevance
                - redundancy_weight * redundancy
                + type_diversity_weight * type_bonus
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        if best_index is None:
            break

        selected.append(best_index)
        selected_types.add(vle.iloc[best_index]["activity_type"])
        remaining.remove(best_index)

    recommendations = []
    for index in selected:
        recommendations.append({
            "id_site": int(vle.iloc[index]["id_site"]),
            "activity_type": str(vle.iloc[index]["activity_type"]),
            "recommendation_score": round(float(scores[index]), 4),
        })

    return recommendations


def generate_recommendations(
    student_id,
    datasets,
    top_n=5,
    decay_rate=0.1,
    interest_weight=0.1,
):
    """
    Hybrid personalized recommendation system.

    Combines:
    1. Student interest similarity (TF-IDF + Cosine Similarity)
    2. Exponential time-decay weighting on recent clicks
    3. Log-normalized course resource popularity
    4. MMR diversity ranking
    """
    if not 0 <= interest_weight <= 1:
        raise ValueError("interest_weight must be between 0 and 1")

    if decay_rate < 0:
        raise ValueError("decay_rate cannot be negative")

    if "student_vle" not in datasets or "vle" not in datasets:
        raise ValueError("Datasets must contain 'student_vle' and 'vle' DataFrames")

    student_vle = datasets["student_vle"].copy()
    vle = datasets["vle"].reset_index(drop=True).copy()

    if student_vle.empty or vle.empty:
        raise ValueError("Input datasets cannot be empty")

    # Ensure required defensive columns exist for text_processor compatibility
    if "is_evergreen" not in vle.columns:
        if "week_from" in vle.columns and "week_to" in vle.columns:
            vle["is_evergreen"] = (
                vle["week_from"].isna() & vle["week_to"].isna()
            ).astype(int)
        else:
            vle["is_evergreen"] = 0

    student_vle["id_student"] = pd.to_numeric(
        student_vle["id_student"], errors="coerce"
    )
    student_vle["date"] = pd.to_numeric(
        student_vle["date"], errors="coerce"
    ).fillna(0)
    student_vle["sum_click"] = pd.to_numeric(
        student_vle["sum_click"], errors="coerce"
    ).fillna(0)

    activity = student_vle[
        student_vle["id_student"] == int(student_id)
    ].copy()

    if activity.empty:
        raise ValueError(f"No activity found for student ID {student_id}")

    # Find the student's most active course presentation
    course = (
        activity.groupby(["code_module", "code_presentation"])["sum_click"]
        .sum()
        .idxmax()
    )
    module, presentation = course

    activity = activity[
        (activity["code_module"] == module)
        & (activity["code_presentation"] == presentation)
    ].copy()

    course_vle = vle[
        (vle["code_module"] == module)
        & (vle["code_presentation"] == presentation)
    ].reset_index(drop=True)

    if course_vle.empty:
        raise ValueError(f"No resources found for course {module} ({presentation})")

    # Build TF-IDF feature matrix for course resources
    processed = prepare_resource_features(course_vle)
    _, tfidf_matrix = build_tfidf_model(processed)

    # Aggregate student click history per resource
    history = (
        activity.groupby("id_site")
        .agg(
            total_clicks=("sum_click", "sum"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    resource_positions = {
        resource_id: index
        for index, resource_id in enumerate(course_vle["id_site"])
    }

    history = history[history["id_site"].isin(resource_positions)].copy()

    if history.empty:
        raise ValueError("No matching student resources found in course VLE")

    history["vector_index"] = history["id_site"].map(resource_positions)

    # Calculate exponential time-decay weights (recent interactions carry higher weight)
    current_day = history["last_date"].max()
    age_in_weeks = np.maximum(0.0, (current_day - history["last_date"]) / 7.0)
    time_weights = np.exp(-decay_rate * age_in_weeks)

    click_weights = (
        history["total_clicks"].to_numpy(dtype=float)
        * time_weights.to_numpy(dtype=float)
    )

    # Fallback to uniform weights if total clicks are zero
    if click_weights.sum() <= 0:
        click_weights = np.ones(len(history), dtype=float)

    # Build weighted student interest vector
    indices = history["vector_index"].to_numpy()
    student_vectors = tfidf_matrix[indices].toarray()

    student_interest = np.average(
        student_vectors,
        axis=0,
        weights=click_weights,
    )

    # Compute content similarity scores across all course resources
    interest_scores = cosine_similarity(
        tfidf_matrix,
        student_interest.reshape(1, -1),
    ).flatten()

    # Compute course-wide resource popularity up to student's current day
    course_activity = student_vle[
        (student_vle["code_module"] == module)
        & (student_vle["code_presentation"] == presentation)
        & (student_vle["date"] <= current_day)
    ]

    popularity = course_activity.groupby("id_site")["sum_click"].sum()

    popularity_scores = (
        course_vle["id_site"]
        .map(popularity)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    # Log-normalize popularity to reduce popularity bias
    popularity_scores = np.log1p(np.maximum(0.0, popularity_scores))
    max_popularity = popularity_scores.max()
    if max_popularity > 0:
        popularity_scores /= max_popularity

    # Hybrid score combining personalized interest and normalized popularity
    hybrid_scores = (
        interest_weight * interest_scores
        + (1.0 - interest_weight) * popularity_scores
    )

    # Prefer unvisited resources; fall back to all course resources if all were visited
    visited_ids = set(history["id_site"])
    candidate_indices = [
        index
        for index, resource_id in enumerate(course_vle["id_site"])
        if resource_id not in visited_ids
    ]

    if not candidate_indices:
        candidate_indices = list(range(len(course_vle)))

    recommendations = diversify_recommendations(
        candidate_indices=candidate_indices,
        scores=hybrid_scores,
        tfidf_matrix=tfidf_matrix,
        vle=course_vle,
        top_n=top_n,
    )

    return recommendations


if __name__ == "__main__":
    from ml.data_loader import load_datasets

    datasets = load_datasets()
    student_id = int(datasets["student_vle"]["id_student"].iloc[0])

    print("\nHYBRID PERSONALIZED RECOMMENDATION SYSTEM")
    print("Student ID:", student_id)

    recommendations = generate_recommendations(
        student_id,
        datasets,
        top_n=5,
    )

    print("\nRecommended resources:")
    for item in recommendations:
        print(item)

    print("\nTotal recommendations:", len(recommendations))