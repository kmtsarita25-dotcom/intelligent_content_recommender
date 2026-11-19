from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "clean"

# TF-IDF configuration
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"

# Recommendation configuration
TOP_N_RECOMMENDATIONS = 5

# Time decay configuration
DECAY_RATE = 0.1

# Preference drift configuration
RECENT_WEEKS = 4

# Evaluation configuration
EVALUATION_K = 5

# Dataset file names
DATA_FILES = {
    "student_vle": "studentVle_cleaned.csv",
    "vle": "vle_cleaned.csv",
    "assessments": "assessments_cleaned.csv",
    "courses": "courses_cleaned.csv",
    "student_assessment": "studentAssessment_cleaned.csv",
    "student_info": "studentInfo_cleaned.csv",
    "student_registration": "studentRegistration_cleaned.csv"
}

if __name__ == "__main__":
    print("\nML CONFIGURATION LOADED")
    print("Project directory:", BASE_DIR)
    print("Dataset directory:", DATA_DIR)
    print("Top recommendations:", TOP_N_RECOMMENDATIONS)
    print("Decay rate:", DECAY_RATE)
    print("Evaluation K:", EVALUATION_K)