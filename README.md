# SpotifyCares AI Support Assistant 🎵🤖

An explainable, retrieval-augmented, and locally executable AI Support Assistant for **SpotifyCares** (`@SpotifyCares`) built for the **Hiver SDE Intern**

The assistant processes incoming customer messages and returns structured JSON:
```json
{
  "intent": "Subscription & Premium",
  "reply": "Hey! Try logging out and logging back in—that usually syncs your payment status right away. DM us your receipt if it stays Free!",
  "decision": "Auto Handle",
  "reason": "General subscription plan status inquiry handled via standard account synchronization steps."
}
```

---

## 🌟 Key Features & Constraints Adherence

- **Zero Fine-Tuning / No Multi-Agent**: 100% local, explainable execution.
- **Multilevel Intent Classifiers**: Baseline 1 (Majority Class), Baseline 2 (TF-IDF + LogReg), and Final Model (SentenceTransformer `all-MiniLM-L6-v2` + LogReg).
- **Local FAISS Retrieval Engine**: Sub-millisecond similarity search over historical `@SpotifyCares` Q&A pairs.
- **Deterministic Rule Escalation Engine**: Transparent business logic for `Auto Handle` vs `Escalate`.
- **Local Reply Quality Evaluation**: 40-sample benchmark using deterministic scoring heuristics for Correctness, Helpfulness, Tone, and Consistency.

---

## 📂 Dataset Source

Dataset:
Customer Support on Twitter (Kaggle)

Dataset URL:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The original dataset contains customer support conversations from multiple companies. For this assignment, SpotifyCares conversations were filtered and processed to create a Spotify-focused support dataset.

Processing Steps:
1. Extract Spotify-related conversations.
2. Remove incomplete threads.
3. Clean URLs, mentions, and noise.
4. Build customer-agent interaction pairs.
5. Create a 200-sample golden evaluation dataset with manual intent labels.

## 📊 Benchmark Summary Results

### 1. Intent Classification Performance (200 Golden Samples)
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Most Frequent** | 12.50% | 0.0156 | 0.1250 | 0.0278 |
| **Baseline 2: TF-IDF + LogReg** | 57.50% | 0.6254 | 0.5750 | 0.5691 |
| **Final: Sentence Transformer + LogReg** | **75.50%** | **0.7671** | **0.7550** | **0.7477** |

### 2. Reply Quality Evaluation (40 Representative Samples)

The evaluation framework uses a lightweight local scoring engine that evaluates generated replies on:

- Correctness  | Average Score: 5.00/5
- Helpfulness  | Average Score: 4.78/5
- Tone         | Average Score: 5.00/5
- Consistency  | Average Score: 5.00/5

This enables fully reproducible offline evaluation without requiring API credits.

---
### Generated Evaluation Artifacts

The following files are generated automatically when running the evaluation scripts and are not stored in the repository:

- evaluation/intent_metrics_summary.csv
- evaluation/reply_quality_evaluation_40.csv
- evaluation/reply_evaluation_summary.csv
- evaluation/failure_analysis_5.csv
- evaluation/failure_analysis_5.md

To reproduce these artifacts:

  ```bash
python -m evaluation.evaluate_intents
python -m evaluation.evaluate_replies
python -m evaluation.failure_analysis
```

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 1. Installation
```bash
git clone https://github.com/aman7827/spotify_ai_support_assistant.git
cd spotify_ai_support_assistant
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python -m src.data_prep
```

### 3. Run Intent Classification Evaluation
```bash
python -m evaluation.evaluate_intents
```

### 4. Run Reply Quality Evaluation
```bash
python -m evaluation.evaluate_replies
```

### 5. Run Failure Analysis
```bash
python -m evaluation.failure_analysis
```

### 6. Run End-to-End Pipeline CLI
```bash
# Automated Test Suite
python -m src.pipeline --test

# Single Customer Message Query
python -m src.pipeline --message "I can't log into my account on iPhone, says invalid password."
```

---

## 📁 Repository Structure

```
spotify_ai_support_assistant/
├── data/
│   ├── spotify_raw_pairs.csv          # SpotifyCares conversations extracted from Kaggle Customer Support on Twitter dataset
│   ├── spotify_cleaned_pairs.csv      # Cleaned SpotifyCares customer-support pairs dataset (537 pairs)
│   └── golden_evaluation_200.csv      # 200 human-labelled customer messages (Intent labels only)
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   # Preprocessing & noise cleaning pipeline
│   ├── intent_classifier.py           # Baseline 1, Baseline 2, & Final ST+LogReg classifier
│   ├── case_retrieval.py              # SentenceTransformers + FAISS vector search
│   ├── reply_generator.py             # Support reply generation with template-based fallback responses
│   ├── escalation_engine.py           # Deterministic business rule escalation engine
│   └── pipeline.py                    # End-to-end SupportAssistant coordinator CLI
├── evaluation/
│   ├── evaluate_intents.py            # Intent classification metrics evaluator
│   ├── evaluate_replies.py            # Local reply quality evaluation framework
│   └── failure_analysis.py            # 5 concrete real failure cases breakdown
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb
│   ├── 02_intent_classification.ipynb
│   ├── 03_retrieval_and_escalation.ipynb
│   └── 04_evaluation_and_failure_analysis.ipynb
├── report/
│   └── Mandatory_Report.md            # Comprehensive 6-page engineering report
├── Decision_Log.md                    # 12 engineering decisions & trade-offs
├── README.md                          # Reproduction guide & documentation
└── requirements.txt                   # Dependency list
```
