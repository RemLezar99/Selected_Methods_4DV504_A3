"""
Small script that lets you inspect the results of the training and important metrics

"""

import pandas as pd

NER_RESULTS_PATH = "results/ner_results_with_lr.csv"
SENTIMENT_RESULTS_PATH = "results/sentiment_results.csv"

NER_RESULTS_ROUNDED_PATH = "results/ner_results_with_lr_rounded.csv"
SENTIMENT_RESULTS_ROUNDED_PATH = "results/sentiment_results_rounded.csv"


def inspect_and_save_ner_results(rounding=4):
    print("\n=== NER RESULTS ===")
    df = pd.read_csv(NER_RESULTS_PATH)

    metric_cols = ["Precision", "Recall", "F1"]
    if rounding is not None:
        for col in metric_cols:
            if col in df.columns:
                df[col] = df[col].round(rounding)

    # Save rounded results for LaTeX
    df.to_csv(NER_RESULTS_ROUNDED_PATH, index=False)
    print(f"\nRounded NER results saved to: {NER_RESULTS_ROUNDED_PATH}")

    print("\nFull table:")
    print(df)

    print("\nBest F1 per model and tagging scheme:")
    best = (
        df.sort_values("F1", ascending=False)
          .groupby(["Model", "Tagging Scheme"], as_index=False)
          .first()
    )
    print(best)

    best_overall = df.loc[df["F1"].idxmax()]
    print("\nBest overall NER setup:")
    print(best_overall)

    print("\nAverage F1 by tagging scheme:")
    print(df.groupby("Tagging Scheme")["F1"].mean())


def inspect_and_save_sentiment_results(rounding=4):
    print("\n=== SENTIMENT RESULTS ===")
    df = pd.read_csv(SENTIMENT_RESULTS_PATH)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    if rounding is not None:
        for col in metric_cols:
            if col in df.columns:
                df[col] = df[col].round(rounding)

    # Save rounded results for LaTeX
    df.to_csv(SENTIMENT_RESULTS_ROUNDED_PATH, index=False)
    print(f"\nRounded sentiment results saved to: {SENTIMENT_RESULTS_ROUNDED_PATH}")

    print("\nFull table:")
    print(df)

    print("\nSorted by F1:")
    print(df.sort_values("F1", ascending=False))


if __name__ == "__main__":
    inspect_and_save_ner_results(rounding=4)
    inspect_and_save_sentiment_results(rounding=4)
