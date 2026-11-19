from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(tfidf_matrix):
    """Calculate similarity between learning resources."""

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def get_similar_resources(
    similarity_matrix,
    vle_df,
    site_id,
    top_n=5
):
    """Find the most similar learning resources."""

    if site_id not in vle_df["id_site"].values:
        raise ValueError(f"Resource ID {site_id} not found")

    index = vle_df.index[
        vle_df["id_site"] == site_id
    ][0]

    scores = list(enumerate(similarity_matrix[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in scores:

        if idx == index:
            continue

        results.append({
            "id_site": int(vle_df.iloc[idx]["id_site"]),
            "activity_type": vle_df.iloc[idx]["activity_type"],
            "similarity_score": round(float(score), 4)
        })

        if len(results) == top_n:
            break

    return results

if __name__ == "__main__":

    from ml.data_loader import load_datasets
    from ml.text_processor import prepare_resource_features
    from ml.tfidf_model import build_tfidf_model

    datasets = load_datasets()

    vle = datasets["vle"].reset_index(drop=True)

    processed = prepare_resource_features(vle)

    vectorizer, tfidf_matrix = build_tfidf_model(processed)

    similarity_matrix = calculate_similarity(tfidf_matrix)

    print("\nCOSINE SIMILARITY COMPLETED")

    print("Similarity matrix shape:", similarity_matrix.shape)

    sample_id = int(vle.iloc[0]["id_site"])

    recommendations = get_similar_resources(
        similarity_matrix,
        vle,
        sample_id,
        top_n=5
    )

    print("\nSimilar resources for:", sample_id)

    for item in recommendations:
        print(item)