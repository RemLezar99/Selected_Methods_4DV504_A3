"""
Small script that lets you inspect the results of the training and important metrics

"""

import pandas as pd

NER_RESULTS_PATH = "results/ner_results_with_lr.csv"
SENTIMENT_RESULTS_PATH = "results/sentiment_results.csv"


def inspect_ner_results():
    print("\n=== NER RESULTS ===")
    df = pd.read_csv(NER_RESULTS_PATH)

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


def inspect_sentiment_results():
    print("\n=== SENTIMENT RESULTS ===")
    df = pd.read_csv(SENTIMENT_RESULTS_PATH)

    print("\nFull table:")
    print(df)

    print("\nSorted by F1:")
    print(df.sort_values("F1", ascending=False))


if __name__ == "__main__":
    inspect_ner_results()
    inspect_sentiment_results()