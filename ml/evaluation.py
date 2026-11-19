import numpy as np
import pandas as pd

from pathlib import Path

from ml.data_loader import load_datasets
from ml.recommender import generate_recommendations


def calculate_metrics(recommended, relevant, k=5):

    recommended = recommended[:k]
    hits = len(set(recommended) & relevant)

    precision = hits / k if k else 0
    recall = hits / len(relevant) if relevant else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return precision, recall, f1


def popularity_recommendations(
    datasets, module, presentation,
    cutoff_date, seen_resources, k=5
):

    student_vle = datasets["student_vle"]

    history = student_vle[
        (student_vle["date"] < cutoff_date) &
        (student_vle["code_module"] == module) &
        (student_vle["code_presentation"] == presentation)
    ]

    popularity = (
        history.groupby("id_site")["sum_click"]
        .sum()
        .sort_values(ascending=False)
    )

    course_resources = set(
        datasets["vle"][
            (datasets["vle"]["code_module"] == module) &
            (datasets["vle"]["code_presentation"] == presentation)
        ]["id_site"]
    )

    recommended = [
        int(site_id)
        for site_id in popularity.index
        if site_id not in seen_resources
        and site_id in course_resources
    ]

    return recommended[:k]


def evaluate_model(max_students=500, k=5, seed=42):

    datasets = load_datasets()
    student_vle = datasets["student_vle"]

    student_ids = (
        student_vle["id_student"]
        .drop_duplicates()
        .sample(
            n=min(max_students, student_vle["id_student"].nunique()),
            random_state=seed
        )
        .tolist()
    )

    personalized_results = []
    popularity_results = []

    personalized_items = set()
    popularity_items = set()

    total_resources = datasets["vle"]["id_site"].nunique()

    evaluated = 0

    for student_id in student_ids:

        student_data = student_vle[
            student_vle["id_student"] == student_id
        ].copy()

        dates = sorted(student_data["date"].dropna().unique())

        if len(dates) < 5:
            continue

        split_index = int(len(dates) * 0.8)
        split_index = max(1, min(split_index, len(dates) - 1))

        cutoff_date = dates[split_index]

        train_data = student_data[
            student_data["date"] < cutoff_date
        ]

        test_data = student_data[
            student_data["date"] >= cutoff_date
        ]

        if train_data.empty or test_data.empty:
            continue

        main_course = (
            train_data.groupby(
                ["code_module", "code_presentation"]
            )["sum_click"]
            .sum()
            .idxmax()
        )

        module, presentation = main_course

        train_course = train_data[
            (train_data["code_module"] == module) &
            (train_data["code_presentation"] == presentation)
        ]

        test_course = test_data[
            (test_data["code_module"] == module) &
            (test_data["code_presentation"] == presentation)
        ]

        seen = set(train_course["id_site"].unique())

        relevant = set(test_course["id_site"].unique()) - seen

        if not relevant:
            continue

        train_datasets = dict(datasets)

        train_datasets["student_vle"] = student_vle[
            (student_vle["id_student"] != student_id) |
            (student_vle["date"] < cutoff_date)
        ].copy()

        try:
            personalized = generate_recommendations(
                student_id,
                train_datasets,
                top_n=k,
                interest_weight=0.1
            )
        except (ValueError, KeyError):
            continue

        personalized_ids = [
            item["id_site"] for item in personalized
        ]

        popular_ids = popularity_recommendations(
            train_datasets,
            module,
            presentation,
            cutoff_date,
            seen,
            k
        )

        p1, r1, f1 = calculate_metrics(
            personalized_ids, relevant, k
        )

        p2, r2, f2 = calculate_metrics(
            popular_ids, relevant, k
        )

        personalized_results.append((p1, r1, f1))
        popularity_results.append((p2, r2, f2))

        personalized_items.update(personalized_ids)
        popularity_items.update(popular_ids)

        evaluated += 1

        if evaluated % 50 == 0:
            print(f"Evaluated {evaluated} students...")

    if evaluated == 0:
        print("No eligible students found.")
        return

    print("\nPOPULARITY VS PERSONALIZED RECOMMENDATION")
    print("------------------------------------------")

    report_rows = []

    for name, results, items in [
        ("Personalized Model", personalized_results,
         personalized_items),
        ("Popularity Baseline", popularity_results,
         popularity_items)
    ]:

        averages = np.mean(results, axis=0)

        coverage = (
            len(items) / total_resources
            if total_resources else 0
        )

        print(f"\n{name}")
        print(f"Precision@{k}: {averages[0]:.4f}")
        print(f"Recall@{k}: {averages[1]:.4f}")
        print(f"F1@{k}: {averages[2]:.4f}")
        print(f"Catalog Coverage: {coverage:.4f}")

        report_rows.append({
            "Model": name,
            f"Precision@{k}": averages[0],
            f"Recall@{k}": averages[1],
            f"F1@{k}": averages[2],
            "Catalog Coverage": coverage,
            "Students Evaluated": evaluated,
            "Interest Weight": 0.1 if name == "Personalized Model" else 0.0,
            "K": k,
            "Random Seed": seed
        })

    print("\nStudents evaluated:", evaluated)

    # Save final validation report
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(report_rows)

    report_path = output_dir / "final_validation_results.csv"
    report.to_csv(report_path, index=False)

    print("\nFinal validation report saved successfully!")
    print("File:", report_path)


if __name__ == "__main__":
    evaluate_model(max_students=500, k=5, seed=42)