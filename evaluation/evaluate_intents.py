"""
evaluation/evaluate_intents.py
------------------------------
Evaluates Baseline 1, Baseline 2, and Final Intent Classification models
against the 200-sample Golden Evaluation Dataset.
Computes Accuracy, Precision, Recall, and F1 Score (Macro & Weighted).
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.intent_classifier import (
    MostFrequentClassifier,
    TfidfLogRegClassifier,
    SentenceTransformerClassifier
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_evaluation_200.csv")
TRAIN_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spotify_cleaned_pairs.csv")

def evaluate_models():
    print("Loading Golden Evaluation Dataset (200 records)...")
    golden_df = pd.read_csv(DATA_PATH)
    train_df = pd.read_csv(TRAIN_CORPUS_PATH)

    # Train corpus (Historical dataset)
    X_train = train_df["customer_message"].tolist()
    y_train = train_df["intent_category"].tolist()

    # Evaluation target (Golden dataset)
    X_test = golden_df["customer_message"].tolist()
    y_test = golden_df["intent"].tolist()

    models = {
        "Baseline 1: Most Frequent": MostFrequentClassifier(),
        "Baseline 2: TF-IDF + LogReg": TfidfLogRegClassifier(),
        "Final: Sentence Transformer + LogReg": SentenceTransformerClassifier()
    }

    results = []

    for name, model in models.items():
        print(f"Fitting & Evaluating: {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0
            )
        p_weight, r_weight, f1_weight, _ = \
            precision_recall_fscore_support(
            y_test,
            preds,
            average="weighted",
            zero_division=0
            )

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(p_macro, 4),
            "Recall (Macro)": round(r_macro, 4),
            "F1 Score (Macro)": round(f1_macro, 4),
            "F1 Score (Weighted)": round(f1_weight, 4)
        })

    df_results = pd.DataFrame(results)
    print("\n================ INTENT CLASSIFICATION EVALUATION METRICS ================")
    print(df_results.to_string(index=False))
    print("==========================================================================\n")

    output_path = os.path.join(os.path.dirname(__file__), "intent_metrics_summary.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Saved evaluation metrics summary to {output_path}")
    return df_results

if __name__ == "__main__":
    evaluate_models()
