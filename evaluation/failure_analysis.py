"""
evaluation/failure_analysis.py
------------------------------

Generates failure analysis from actual evaluation outputs.

Source:
evaluation/reply_quality_evaluation_40.csv

Output:
- failure_analysis_5.csv
- failure_analysis_5.md
"""

import os
import pandas as pd

EVAL_FILE = os.path.join(
    os.path.dirname(__file__),
    "reply_quality_evaluation_40.csv"
)

OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "failure_analysis_5.csv"
)

OUTPUT_MD = os.path.join(
    os.path.dirname(__file__),
    "failure_analysis_5.md"
)


def build_root_cause(row):
    """
    Simple explainable root cause generation.
    """

    if row["golden_intent"] != row["predicted_intent"]:
        return (
            "Intent classifier predicted a different intent "
            "than the golden label."
        )

    return (
        "Generated reply quality was lower than expected "
        "for this query."
    )


def build_fix(row):
    """
    Simple fix suggestion.
    """

    if row["golden_intent"] != row["predicted_intent"]:
        return (
            "Add more training examples for this intent "
            "and review ambiguous intent boundaries."
        )

    return (
        "Improve retrieval quality and provide stronger "
        "historical support examples."
    )


def generate_failure_analysis():

    print("Loading evaluation results...")

    if not os.path.exists(EVAL_FILE):
        raise FileNotFoundError(
            f"Evaluation file not found:\n{EVAL_FILE}\n\n"
            "Run evaluate_replies.py first."
        )

    df = pd.read_csv(EVAL_FILE)

    failures = []

    # Intent misclassification failures
    misclassified = df[
        df["golden_intent"] != df["predicted_intent"]
    ]

    for _, row in misclassified.iterrows():

        failures.append({
            "customer_message": row["customer_message"],
            "expected_intent": row["golden_intent"],
            "actual_intent": row["predicted_intent"],
            "actual_reply": row["generated_reply"],
            "root_cause": build_root_cause(row),
            "proposed_fix": build_fix(row)
        })

    # If fewer than 5 failures found,
    # fill remaining slots using lowest LLM scores

    if len(failures) < 5:

        df["avg_score"] = (
            df["llm_correctness"] +
            df["llm_helpfulness"] +
            df["llm_tone"] +
            df["llm_consistency"]
        ) / 4

        low_score_cases = df.sort_values(
            by="avg_score"
        )

        for _, row in low_score_cases.iterrows():

            if len(failures) >= 5:
                break

            failures.append({
                "customer_message": row["customer_message"],
                "expected_intent": row["golden_intent"],
                "actual_intent": row["predicted_intent"],
                "actual_reply": row["generated_reply"],
                "root_cause": build_root_cause(row),
                "proposed_fix": build_fix(row)
            })

    failures = failures[:5]

    for i, failure in enumerate(failures, start=1):
        failure["failure_id"] = f"FAIL-{i:03d}"

    df_failures = pd.DataFrame(failures)

    df_failures.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# SpotifyCares AI Support Assistant - Failure Analysis\n\n"
        )

        for _, row in df_failures.iterrows():

            f.write(
                f"## {row['failure_id']}\n\n"
            )

            f.write(
                f"**Customer Message**\n\n"
                f"{row['customer_message']}\n\n"
            )

            f.write(
                f"**Expected Intent:** "
                f"{row['expected_intent']}\n\n"
            )

            f.write(
                f"**Predicted Intent:** "
                f"{row['actual_intent']}\n\n"
            )

            f.write(
                f"**Generated Reply**\n\n"
                f"{row['actual_reply']}\n\n"
            )

            f.write(
                f"**Root Cause**\n\n"
                f"{row['root_cause']}\n\n"
            )

            f.write(
                f"**Proposed Fix**\n\n"
                f"{row['proposed_fix']}\n\n"
            )

            f.write("---\n\n")

    print(
        f"Saved:\n"
        f" - {OUTPUT_CSV}\n"
        f" - {OUTPUT_MD}"
    )

    return df_failures


if __name__ == "__main__":
    generate_failure_analysis()
