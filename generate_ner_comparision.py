import pandas as pd

# Paths
NER_RESULTS_ROUNDED_PATH = "results/ner_results_with_lr_rounded.csv"
LR_COMPARISON_PATH = "results/ner_lr_comparison.csv"

# Extension setup (explicit and controlled)
MODEL_NAME = "google-bert/bert-base-multilingual-cased"
TAGGING_SCHEME = "BIO"


def generate_lr_comparison():
    # Load rounded NER results
    df = pd.read_csv(NER_RESULTS_ROUNDED_PATH)

    # Filter to chosen model and tagging scheme
    df_filtered = df[
        (df["Model"] == MODEL_NAME) &
        (df["Tagging Scheme"] == TAGGING_SCHEME)
    ]

    if df_filtered.empty:
        raise ValueError(
            "No matching rows found for the specified model and tagging scheme."
        )

    # Select relevant columns
    lr_df = df_filtered[
        ["Learning Rate", "Precision", "Recall", "F1"]
    ].sort_values("Learning Rate")

    # Save CSV for LaTeX
    lr_df.to_csv(LR_COMPARISON_PATH, index=False)

    print("Learning-rate comparison table created:")
    print(lr_df)
    print(f"\nSaved to: {LR_COMPARISON_PATH}")


if __name__ == "__main__":
    generate_lr_comparison()