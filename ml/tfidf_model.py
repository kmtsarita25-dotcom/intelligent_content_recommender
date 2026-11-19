from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_model(df):
    """Convert learning resource text into TF-IDF vectors."""

    if "item_text" not in df.columns:
        raise ValueError("item_text column not found")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["item_text"].fillna("")
    )

    return vectorizer, tfidf_matrix

if __name__ == "__main__":

    from ml.data_loader import load_datasets
    from ml.text_processor import prepare_resource_features

    datasets = load_datasets()

    vle = datasets["vle"]

    processed = prepare_resource_features(vle)

    vectorizer, tfidf_matrix = build_tfidf_model(processed)

    print("\nTF-IDF MODEL CREATED SUCCESSFULLY")

    print("Number of resources:", tfidf_matrix.shape[0])
    print("Number of features:", tfidf_matrix.shape[1])

    print("\nVocabulary:")
    print(vectorizer.get_feature_names_out())