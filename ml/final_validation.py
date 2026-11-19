import pandas as pd

from ml.evaluation import evaluate_model

def final_validation():

    print("\n" + "=" * 55)
    print("       FINAL ML MODEL VALIDATION")
    print("=" * 55)

    print("\nSelected configuration:")
    print("Interest weight: 0.1")
    print("Relevance weight: 0.85")
    print("Type diversity weight: 0.00")

    print("\nEvaluation settings:")
    print("Students: Up to 500")
    print("Actual evaluated students: Expected 444")
    print("Top-K recommendations: 5")
    print("Random seed: 42")

    print("\nStarting final evaluation...\n")

    evaluate_model(
        max_students=500,
        k=5,
        seed=42
    )

    print("\n" + "=" * 55)
    print("       FINAL VALIDATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    final_validation()
