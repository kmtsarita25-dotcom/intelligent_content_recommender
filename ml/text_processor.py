import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def prepare_resource_features(vle_df):

    df = vle_df.copy()

    text_columns = [
        "activity_type",
        "code_module",
        "code_presentation"
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .fillna("unknown")
            .astype(str)
        )

    df["week_from"] = pd.to_numeric(
        df["week_from"], errors="coerce"
    ).fillna(0)

    df["week_to"] = pd.to_numeric(
        df["week_to"], errors="coerce"
    ).fillna(0)

    df["is_evergreen"] = pd.to_numeric(
        df["is_evergreen"], errors="coerce"
    ).fillna(0)

    df["item_text"] = (
        "activity " + df["activity_type"] +
        " module " + df["code_module"] +
        " presentation " + df["code_presentation"] +
        " startweek " + df["week_from"].astype(int).astype(str) +
        " endweek " + df["week_to"].astype(int).astype(str) +
        " evergreen " + df["is_evergreen"].astype(int).astype(str)
    )

    df["item_text"] = df["item_text"].apply(clean_text)

    return df

if __name__ == "__main__":

    from ml.data_loader import load_datasets

    datasets = load_datasets()

    processed = prepare_resource_features(
        datasets["vle"]
    )

    print("\nIMPROVED RESOURCE FEATURES COMPLETED")
    print("Total resources:", len(processed))
    print("\nSample features:")

    print(
        processed[
            ["id_site", "activity_type", "week_from",
             "week_to", "item_text"]
        ].head(5).to_string(index=False)
    )