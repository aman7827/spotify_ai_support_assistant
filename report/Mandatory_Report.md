- Technical Report: SpotifyCares AI Support Assistant

**Candidate**: Candidate / SDE Intern Applicant (2027 Batch)
**Target Brand**: SpotifyCares (`@SpotifyCares`)
**Domain**: Support Automation & Intent Triage

---

1. Problem Framing

Customer support accounts like @SpotifyCares receive thousands of tweets every day. These messages range from urgent billing bugs and hacked accounts to simple playback glitches and general feature questions.

Manually reading and tagging every single tweet creates long wait times for users and risks missing urgent security issues.

Instead of building an open-ended chatbot that might hallucinate replies, this project treats support automation as a structured 5-stage pipeline:
$$\text{Customer Message} \longrightarrow \text{Intent Classification} \longrightarrow \text{Case Retrieval} \longrightarrow \text{Reply Generation} \longrightarrow \text{Escalation Decision}$$


- The system outputs a clear JSON payload for every incoming message:

{
  "intent": "Subscription Issue",
  "reply": "Hey! Try logging out and logging back in to sync your payment status...",
  "decision": "Auto Handle ",
  "reason": "Standard account sync steps applied for active subscription query."
}
```

The system prioritizes **predictability, explainability, and local auditability** over black-box model sophistication.

---

2. Dataset & Preprocessing
This project uses historical Twitter support data filtered specifically for SpotifyCares.

Why SpotifyCares?
Diverse Customer Issues: Covers clear categories like payment errors, subscription bugs, app crashes, and login failures.

Distinct Intent Boundaries: Issues naturally group into clean categories compared to standard retail support.

Structured Q&A Format: Twitter interactions follow a quick question → answer structure that works great for retrieval-augmented support systems.

Text Cleaning Steps
To keep the preprocessing simple and easy to debug:

Tag Removal: Removed @SpotifyCares and user handles.

Agent Sign-off Removal: Stripped Twitter agent signatures (e.g., ^NK, ^JM).

URL Sanitization: Replaced links with a generic [URL] tag.

Length Filtering: Dropped short noise tweets under 10 characters.

Golden Benchmark Dataset
A balanced 200-sample golden evaluation dataset (data/golden_evaluation_200.csv) was manually created, containing exactly 25 labeled examples across 8 intent categories.
---

3. System Design & Architecture

```
[Customer Message]
       ▼
│  Stage 1: Intent Classifier  │  (SentenceTransformer embeddings + Logistic Regression)

               │ (Predicted Intent)

│  Stage 2: FAISS Case Retrieval│ (Dense Cosine Similarity top-3 historical Q&A search)

               │ (Top-3 Cases)

│ Stage 3: LLM Reply Generator │ (Short prompt generation matching Spotify tone)

               │ (Draft Reply)

│ Stage 4: Escalation Engine   │ (Deterministic business rule risk engine)

               │

│ Stage 5: Structured JSON Output

```

---

4. Models Tested for Intent Classification
We compared three baseline approaches on the 200-sample Golden Evaluation Dataset:
1. Baseline 1: Most Frequent Class (Majority Vote)Always predicts the most common class. Serves as our absolute minimum benchmark ($1/8 = 12.5\%$).

2. Baseline 2: TF-IDF + Logistic RegressionExtracts unigram and bigram TF-IDF features (max 1,000 features) and trains a multi-class Logistic Regression classifier.

3. Final Model: Sentence Transformer (all-MiniLM-L6-v2) + Logistic RegressionConverts customer queries into 384-dimensional dense semantic vectors using all-MiniLM-L6-v2, then classifies them with Logistic Regression. No deep neural network fine-tuning was required.
---

5. Evaluation Results

- Intent Classification Benchmark (200 Golden Samples)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Most Frequent** | 0.1250 | 0.0156 | 0.1250 | 0.0278 | 0.0278 |
| **Baseline 2: TF-IDF + LogReg** | 0.5750 | 0.6254 | 0.5750 | 0.5691 | 0.5691 |
| **Final: Sentence Transformer + LogReg** | **0.7550** | **0.7671** | **0.7550** | **0.7477** | **0.7477** |

- Reply Quality Evaluation (40 Representative Samples)

We scored generated support replies on a 1–5 scale across 4 core dimensions, comparing automated LLM-as-a-Judge ratings against manual human ratings:

| Metric | LLM Judge Avg | Human Evaluator Avg | Score Agreement ($\le 1$ pt) | Pearson Correlation |
| :--- | :---: | :---: | :---: | :---: |
| **Correctness** | 4.75 / 5.0 | 4.60 / 5.0 | **100.0%** | 0.798 |
| **Helpfulness** | 4.47 / 5.0 | 4.42 / 5.0 | **100.0%** | 0.552 |
| **Tone** | 5.00 / 5.0 | 4.83 / 5.0 | **100.0%** | N/A (Low Variance) |
| **Consistency** | 5.00 / 5.0 | 4.83 / 5.0 | **100.0%** | N/A (Low Variance) |

---

- 6. Failure Analysis
We extracted 5 real failure cases from empirical evaluation runs to analyze root causes and engineer future fixes:

FAIL-1: Gift Card Activation Misclassified as Billing Dispute

Message: "Redeemed a 12-month gift card but my account shows only 1 month active."

Expected: Subscription & Premium (Auto Handle) → Actual: Billing & Refund Request (Escalate).

Root Cause: Confusion caused by overlapping monetary terms like "redeemed" and "1 month active".

Fix: Add specific keyword rules for gift cards and voucher redemptions.

2. FAIL-2: Playback Feature Issue Misclassified as App Crash

Message: "Crossfade between tracks isn't working on my iOS app after latest update."

Expected: Playback & Audio Bugs → Actual: App Crash & Performance.

Root Cause: The model focused heavily on "after latest update" instead of "Crossfade".

Fix: Give higher feature weight to audio domain terms over generic update keywords.

3. FAIL-3: Login Restriction Error Misclassified as Security Breach

Message: "Multi-device login restriction error popup popping up unexpectedly."

Expected: Login & Auth Issue → Actual: Account Compromise & Security.

Root Cause: The classifier mistook device warning popups for account hacking attempts.

Fix: Check for specific error codes before routing to security escalation.

4. FAIL-4: Mid-Month Plan Drop Ambiguity

Message: "Why did my Premium status drop back to Free mid-month?"

Expected: Subscription & Premium → Actual: Billing & Refund Request.

Root Cause: Overlap between payment sync steps and billing cancellation cycles.

Fix: Add secondary confidence checks for queries that touch on both payment and subscription status.

5. FAIL-5: Feature Query Retrieval Context Mismatch

Message: "Can I disable podcast recommendations from showing up on my home feed?"

Expected: General Query & Feature Request → Actual: App Crash & Performance.

Root Cause: The FAISS retrieval index pulled UI crash examples instead of feature questions.

Fix: Filter FAISS searches so it only queries historical cases inside the predicted intent category.
---

7. Known Limitations
Dataset Size: While real Q&A pairs were used, expanding dataset variations relied partly on standard templates.

In-Memory FAISS Index: The FAISS index runs locally in memory without a persistent database backend.

Escalation Dependency on Classifier: The escalation logic relies on correct intent prediction. Misclassifying a security query as a general query could bypass human review.

---

8. What Is Misleading About My Headline Number?
Our top intent accuracy is 75.5%. While this is a massive jump from the 12.5% baseline, claiming 75.5% represents "total system accuracy" would be misleading for three reasons:

Balanced Data vs. Real Support Volume: The evaluation set uses an equal split across intents. In real production, Login & Auth and Playback Bugs make up over 60% of all incoming tweets, so live accuracy will differ.

Intent Accuracy ≠ Resolution Accuracy: Correctly classifying an intent as Subscription & Premium (75.5% accurate) does not guarantee a helpful reply if FAISS retrieves an irrelevant historical case (as seen in FAIL-5).

Unequal Cost of Mistakes: A 75.5% accuracy metric treats all errors as equal. However, misclassifying a security breach as a general query is far worse than misclassifying an app crash as a audio glitch.

---

9. What I Would Do With One More Week
If given one more week, I would focus on these practical improvements:

Hybrid Search (BM25 + FAISS): Combine exact keyword search (BM25) with dense vector search to stop retrieval mismatches like FAIL-5.

Low-Confidence Clarification Prompts: If the classifier's confidence score drops below 55%, ask the user a quick multi-choice clarification question instead of guessing.

Intent-Filtered FAISS Search: Restrict vector search so it only looks for similar historical cases inside the predicted intent category.

FastAPI & Simple Frontend: Build a lightweight FastAPI backend (/api/v1/support/process) and a simple React interface to demonstrate how support agents would view and approve automated replies live.
