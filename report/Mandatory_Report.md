- Technical Report: SpotifyCares AI Support Assistant

**Candidate:** Candidate / SDE Intern Applicant (2027 Batch)
**Target Brand:** SpotifyCares (`@SpotifyCares`)
**Domain:** Support Automation & Intent Triage

---

1. Problem Framing

Customer support accounts like `@SpotifyCares` receive thousands of tweets every day. These messages include urgent billing problems, hacked accounts, playback issues, and general feature questions.

Manually reading and tagging every tweet can increase waiting time for users and may cause urgent security issues to be missed.

Instead of building an open-ended chatbot that may generate wrong replies, this project uses a structured 5-stage pipeline:

**Customer Message → Intent Classification → Case Retrieval → Reply Generation → Escalation Decision**

The system gives a clear JSON output for every incoming message:

```json
{
  "intent": "Subscription Issue",
  "reply": "Hey! Try logging out and logging back in to sync your payment status...",
  "decision": "Auto Handle",
  "reason": "Standard account sync steps applied for active subscription query."
}
```

The system focuses on **predictability, explainability, and local auditability** instead of using a black-box model.

---

2. Dataset & Preprocessing

This project uses historical Twitter support data filtered specifically for SpotifyCares.

- Why SpotifyCares?

**Diverse Customer Issues:**
The dataset contains different types of issues such as payment errors, subscription bugs, app crashes, and login failures.

**Distinct Intent Boundaries:**
The issues naturally fit into clear categories compared to standard retail support.

**Structured Q&A Format:**
Twitter conversations usually follow a quick question → answer format, which works well for retrieval-based support systems.

- Text Cleaning Steps

The preprocessing steps are kept simple and easy to debug.

**Tag Removal:**
Removed `@SpotifyCares` and other user handles.

**Agent Sign-off Removal:**
Removed Twitter agent signatures such as `^NK` and `^JM`.

**URL Sanitization:**
Replaced links with a general `[URL]` tag.

**Length Filtering:**
Removed short noise tweets with fewer than 10 characters.
- Golden Benchmark Dataset

A balanced 200-sample golden evaluation dataset (`data/golden_evaluation_200.csv`) was manually created.

It contains exactly **25 labeled examples across each of the 8 intent categories**.

---

3. System Design & Architecture

```text
[Customer Message]
       ▼
│ Stage 1: Intent Classifier │
│ SentenceTransformer +      │
│ Logistic Regression        │
       │
       ▼
(Predicted Intent)
       │
       ▼
│ Stage 2: FAISS Case Retrieval │
│ Dense Cosine Similarity        │
│ Top-3 historical Q&A search    │
       │
       ▼
(Top-3 Cases)
       │
       ▼
│ Stage 3: LLM Reply Generator │
│ Short prompt generation      │
│ matching Spotify tone        │
       │
       ▼
(Draft Reply)
       │
       ▼
│ Stage 4: Escalation Engine │
│ Deterministic business     │
│ rule risk engine           │
       │
       ▼
(Quality Control)
       │
       ▼
│ Stage 5: Structured JSON Output │
```

---

4. Models Tested for Intent Classification

We compared three approaches on the **200-sample Golden Evaluation Dataset**.

1. Baseline 1: Most Frequent Class (Majority Vote)

Always predicts the most common class.

It serves as the minimum benchmark:

**1/8 = 12.5%**

2. Baseline 2: TF-IDF + Logistic Regression

Extracts unigram and bigram TF-IDF features with a maximum of **1,000 features** and trains a multi-class Logistic Regression classifier.

3. Final Model: Sentence Transformer + Logistic Regression

Converts customer queries into **384-dimensional dense semantic vectors** using `all-MiniLM-L6-v2`, then classifies them using Logistic Regression.

No deep neural network fine-tuning was required.

---

5. Evaluation Results

- Intent Classification Benchmark — 200 Golden Samples

| Model                                    |   Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | F1 Score (Weighted) |
| ---------------------------------------- | ---------: | ----------------: | -------------: | ---------------: | ------------------: |
| **Baseline 1: Most Frequent**            |     0.1250 |            0.0156 |         0.1250 |           0.0278 |              0.0278 |
| **Baseline 2: TF-IDF + LogReg**          |     0.5750 |            0.6254 |         0.5750 |           0.5691 |              0.5691 |
| **Final: Sentence Transformer + LogReg** | **0.7550** |        **0.7671** |     **0.7550** |       **0.7477** |          **0.7477** |

---

- Reply Quality Evaluation — 40 Representative Samples

Generated support replies were scored on a **1–5 scale** across 4 main dimensions.

The automated LLM-as-a-Judge ratings were compared with manual human ratings.

| Metric          | LLM Judge Avg | Human Evaluator Avg | Score Agreement (≤ 1 pt) | Pearson Correlation |
| --------------- | ------------: | ------------------: | -----------------------: | ------------------: |
| **Correctness** |    4.75 / 5.0 |          4.60 / 5.0 |               **100.0%** |               0.798 |
| **Helpfulness** |    4.47 / 5.0 |          4.42 / 5.0 |               **100.0%** |               0.552 |
| **Tone**        |    5.00 / 5.0 |          4.83 / 5.0 |               **100.0%** |  N/A (Low Variance) |
| **Consistency** |    5.00 / 5.0 |          4.83 / 5.0 |               **100.0%** |  N/A (Low Variance) |

---

- 6. Failure Analysis

We extracted 5 real failure cases from evaluation runs to understand the main problems and possible fixes.

- FAIL-1: Gift Card Activation Misclassified as Billing Dispute

**Message:**

> "Redeemed a 12-month gift card but my account shows only 1 month active."

**Expected:**
Subscription & Premium → Auto Handle

**Actual:**
Billing & Refund Request → Escalate

**Root Cause:**
The model was confused by monetary terms such as "redeemed" and "1 month active".

**Fix:**
Add specific keyword rules for gift cards and voucher redemptions.

---

- FAIL-2: Playback Feature Issue Misclassified as App Crash

**Message:**

> "Crossfade between tracks isn't working on my iOS app after latest update."

**Expected:**
Playback & Audio Bugs

**Actual:**
App Crash & Performance

**Root Cause:**
The model focused more on "after latest update" instead of "Crossfade".

**Fix:**
Give higher feature weight to audio-related terms instead of general update keywords.

---

- FAIL-3: Login Restriction Error Misclassified as Security Breach

**Message:**

> "Multi-device login restriction error popup popping up unexpectedly."

**Expected:**
Login & Auth Issue

**Actual:**
Account Compromise & Security

**Root Cause:**
The classifier confused device warning popups with account hacking attempts.

**Fix:**
Check for specific error codes before routing the case to security escalation.

---

- FAIL-4: Mid-Month Plan Drop Ambiguity

**Message:**

> "Why did my Premium status drop back to Free mid-month?"

**Expected:**
Subscription & Premium

**Actual:**
Billing & Refund Request

**Root Cause:**
There is overlap between payment sync steps and billing cancellation cycles.

**Fix:**
Add secondary confidence checks for queries that involve both payment and subscription status.

---

- FAIL-5: Feature Query Retrieval Context Mismatch

**Message:**

> "Can I disable podcast recommendations from showing up on my home feed?"

**Expected:**
General Query & Feature Request

**Actual:**
App Crash & Performance

**Root Cause:**
The FAISS retrieval index selected UI crash examples instead of feature-related questions.

**Fix:**
Filter FAISS searches so that it only searches historical cases inside the predicted intent category.

---

7. Known Limitations

- Dataset Size

Real Q&A pairs were used, but expanding dataset variations relied partly on standard templates.

- In-Memory FAISS Index

The FAISS index runs locally in memory without a persistent database backend.

- Escalation Dependency on Classifier

The escalation logic depends on correct intent prediction.

Misclassifying a security query as a general query could bypass human review.

---

8. What Is Misleading About My Headline Number?

Our top intent accuracy is **75.5%**.

While this is a large improvement from the **12.5% baseline**, saying that **75.5% represents "total system accuracy"** would be misleading for three reasons.

- Balanced Data vs. Real Support Volume

The evaluation set uses an equal split across intents.

In real production, Login & Auth and Playback Bugs make up over **60%** of all incoming tweets, so live accuracy will be different.

- Intent Accuracy ≠ Resolution Accuracy

Correctly classifying an intent as Subscription & Premium with 75.5% accuracy does not guarantee a helpful reply if FAISS retrieves an unrelated historical case, as seen in FAIL-5.

- Unequal Cost of Mistakes

A 75.5% accuracy metric treats all errors equally.

However, misclassifying a security breach as a general query is much more serious than misclassifying an app crash as an audio glitch.

---

- 9. What I Would Do With One More Week

If given one more week, I would focus on these practical improvements:

- Hybrid Search (BM25 + FAISS)

Combine exact keyword search using BM25 with dense vector search using FAISS to reduce retrieval mismatches like FAIL-5.

- Low-Confidence Clarification Prompts

If the classifier confidence drops below **55%**, ask the user a quick multiple-choice clarification question instead of guessing.

- Intent-Filtered FAISS Search

Restrict vector search so that it only searches historical cases inside the predicted intent category.

- FastAPI & Simple Frontend

Build a lightweight FastAPI backend:

```text
/api/v1/support/process
```

and a simple React interface to show how support agents can view and approve automated replies live.
