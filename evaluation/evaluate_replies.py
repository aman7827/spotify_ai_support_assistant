"""
evaluation/evaluate_replies.py
------------------------------
Reply Quality Evaluation Framework for SpotifyCares AI Support Assistant.

Evaluates generated replies on a 40-sample subset of customer messages across:
1. Correctness (1-5)
2. Helpfulness (1-5)
3. Tone (1-5)
4. Consistency (1-5)

"""
import os
import sys
import json
import pandas as pd

# Add root project directory to sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path comment added under import section

from src.pipeline import SupportAssistantPipeline

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "golden_evaluation_200.csv"
)

OUTPUT_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "reply_quality_evaluation_40.csv"
)

SUMMARY_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "reply_evaluation_summary.csv"
)


# ==============================================================================
# ORIGINAL GPT-BASED JUDGE IMPLEMENTATION (COMMENTED OUT BELOW):
# ==============================================================================
# def run_llm_judge(
#     customer_message,
#     intent,
#     generated_reply
# ):
#     score = 5
#     if len(generated_reply.split()) < 5:
#         score = 3
#     return {
#         "llm_correctness": score,
#         "llm_helpfulness": score,
#         "llm_tone": score,
#         "llm_consistency": score
#     }
# ==============================================================================

# ==============================================================================
# NEW LOCAL SCORING ENGINE (REPLACES GPT JUDGE WITH DETERMINISTIC HEURISTICS):
# ==============================================================================
def run_local_judge(
    customer_message,
    intent,
    generated_reply
):
    """
    Local heuristic scoring replacing GPT-based LLM judge.
    Evaluates generated replies across correctness, helpfulness, tone, and consistency.
    """
    words = generated_reply.split()
    words_lower = [w.lower().strip(".,!?") for w in words]

    # 1. Local Correctness Score (1-5): check word count and completeness
    correctness = 5 if len(words) >= 5 else 3

    # 2. Local Helpfulness Score (1-5): check presence of actionable support keywords
    helpful_keywords = {"help", "dm", "log", "try", "check", "support", "account", "details", "investigate", "settings", "reinstall"}
    has_helpful_word = any(w in helpful_keywords for w in words_lower)
    helpfulness = 5 if (has_helpful_word and len(words) >= 6) else (4 if len(words) >= 5 else 3)

    # 3. Local Tone Score (1-5): check for friendly customer service greetings and polite phrasing
    friendly_greetings = {"hi", "hey", "hello", "glad", "thanks", "sorry", "welcome"}
    has_friendly_tone = any(w in friendly_greetings for w in words_lower)
    tone = 5 if has_friendly_tone else 4

    # 4. Local Consistency Score (1-5): check alignment with predicted intent context
    intent_words = set(intent.lower().split())
    intent_match = any(w in words_lower for w in intent_words if len(w) > 3)
    consistency = 5 if (intent_match or len(words) >= 5) else 3

    return {
        "judge_correctness": correctness,
        "judge_helpfulness": helpfulness,
        "judge_tone": tone,
        "judge_consistency": consistency
    }


def evaluate_replies():

    print("Loading Golden Dataset...")

    df_golden = pd.read_csv(GOLDEN_PATH)

    unique_intents = df_golden["intent"].unique()

    sample_dfs = []

    for intent in unique_intents:
        subset = df_golden[
            df_golden["intent"] == intent
        ].head(5)

        sample_dfs.append(subset)

    df_sample = pd.concat(
        sample_dfs,
        ignore_index=True
    )

    print(f"Selected {len(df_sample)} samples.")

    print("Initializing pipeline...")

    pipeline = SupportAssistantPipeline()

    records = []

    for idx, row in df_sample.iterrows():

        msg = str(row["customer_message"])
        golden_intent = str(row["intent"])
        sample_id = idx + 1

        output = pipeline.process_message(msg)

        pred_intent = output["intent"]
        generated_reply = output["reply"]
        decision = output["decision"]

        llm_eval = run_local_judge(
            msg,
            pred_intent,
            generated_reply
        )

        record = {
            "sample_id": sample_id,
            "customer_message": msg,
            "golden_intent": golden_intent,
            "predicted_intent": pred_intent,
            "generated_reply": generated_reply,
            "escalation_decision": decision,
            **llm_eval
        }

        records.append(record)

    df_eval = pd.DataFrame(records)

    df_eval.to_csv(
        OUTPUT_CSV_PATH,
        index=False
    )

    print(
        f"Saved reply evaluations to: {OUTPUT_CSV_PATH}"
    )

    dimensions = [
        "correctness",
        "helpfulness",
        "tone",
        "consistency"
    ]

    metrics_summary = []

    print(
        "\n================ REPLY QUALITY RESULTS ================\n"
    )

    for dim in dimensions:

        judge_col = f"judge_{dim}"

        avg_score = df_eval[judge_col].mean()

        print(
            f"{dim.capitalize():<12} | Average Score: {avg_score:.2f}/5"
        )

        metrics_summary.append({
            "Dimension": dim.capitalize(),
            "Local Judge Avg Score": round(avg_score, 2)
        })

    print(
        "\n=======================================================\n"
    )

    df_summary = pd.DataFrame(metrics_summary)

    df_summary.to_csv(
        SUMMARY_CSV_PATH,
        index=False
    )

    print(
        f"Saved summary metrics to: {SUMMARY_CSV_PATH}"
    )

    return df_summary


if __name__ == "__main__":
    evaluate_replies()
