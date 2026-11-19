from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "clean"

# OULAD cleaned dataset files
DATA_FILES = {
    "student_vle": "studentVle_cleaned.csv",
    "vle": "vle_cleaned.csv",
    "assessments": "assessments_cleaned.csv",
    "courses": "courses_cleaned.csv",
    "student_assessment": "studentAssessment_cleaned.csv",
    "student_info": "studentInfo_cleaned.csv",
    "student_registration": "studentRegistration_cleaned.csv"
}

def load_datasets():
    """Load all seven cleaned OULAD CSV files."""

    datasets = {}

    for name, filename in DATA_FILES.items():

        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        datasets[name] = pd.read_csv(file_path)
    return datasets

def inspect_datasets(datasets):
    """Display dataset information."""

    print("\n========== OULAD DATASET REPORT ==========\n")

    for name, df in datasets.items():

        print(f"\nDataset: {name}")
        print("-" * 45)

        print("Rows and columns:", df.shape)

        print("\nColumn names:")
        print(list(df.columns))

        print("\nMissing values:")
        print(df.isnull().sum())

    print("\n========== DATA LOADING COMPLETED ==========")

if __name__ == "__main__":

    data = load_datasets()
    inspect_datasets(data)