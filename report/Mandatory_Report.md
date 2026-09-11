# Technical Report: SpotifyCares AI Support Assistant

**Candidate**: Candidate / SDE Intern Applicant (2027 Batch)  
**Target Brand**: SpotifyCares (`@SpotifyCares`)  
**Domain**: Customer Support Automation & Intent Classification  

---

## 1. Problem Framing

In modern multi-channel customer operations, brand support handles like `@SpotifyCares` receive thousands of inbound tweets daily. These messages range from critical billing errors and account compromises to routine playback questions and general feature inquiries. 

Manually routing and triaging every incoming tweet introduces significant support latency, inflates operational costs, and risks missing time-sensitive security emergencies.

This project frames the customer support automation task not as an uncontrolled generative chat problem, but as a **structured 5-stage decision pipeline**:
$$	ext{Customer Message} \longrightarrow 	ext{Intent Classification} \longrightarrow 	ext{Historical Case Retrieval} \longrightarrow 	ext{Reply Generation} \longrightarrow 	ext{Escalation Decision}$$

The assistant returns a structured JSON payload:
```json
{
  "intent": "Subscription Issue",
  "reply": "Hey! Try logging out and logging back in to sync your payment status...",
  "decision": "Auto Handle | Escalate",
  "reason": "Clear explanation of rule decision"
}
```

The system prioritizes **predictability, explainability, and local auditability** over black-box model sophistication.

---

## 2. Dataset Selection

The assignment utilizes historical Twitter customer support data specifically extracted for **SpotifyCares** (`@SpotifyCares`).

### Why SpotifyCares?
1. **High Volume & High Diversity**: Spotify support covers financial transactions (subscriptions, refunds), security (compromised accounts), technical app issues (audio bugs, crashes), and feature queries.
2. **Clear Intent Boundaries**: Distinct business categories exist with minimal brand ambiguity compared to retail or airline support.
3. **Structured Q&A Pattern**: Twitter interactions follow a concise question $ightarrow$ resolution structure suitable for retrieval-augmented support reply generation.

### Minimal, Explainable Preprocessing
Preprocessing is kept intentionally simple and defensible:
- **Handle Normalization**: Stripped `@SpotifyCares` tags.
- **Agent Sign-off Removal**: Stripped Twitter support agent sign-offs (e.g., `^NK`, `^JM`).
- **URL Sanitization**: Replaced http/https hyperlinks.
- **Length Filtering**: Removed empty or short noise tweets (<10 characters).

### Golden Evaluation Dataset
A dedicated 200-sample hand-labelled evaluation benchmark (`data/golden_evaluation_200.csv`) was created, strictly balanced with 25 examples across each of the 8 business intents.

---

## 3. System Design & Architecture

```
[Customer Message]
       │
       ▼
┌──────────────────────────────┐
│  Stage 1: Intent Classifier  │  (SentenceTransformer embeddings + Logistic Regression)
└──────────────┬───────────────┘
               │ (Predicted Intent)
               ▼
┌──────────────────────────────┐
│  Stage 2: FAISS Case Retrieval│ (Dense Cosine Similarity top-3 historical Q&A search)
└──────────────┬───────────────┘
               │ (Top-3 Cases)
               ▼
┌──────────────────────────────┐
│ Stage 3: LLM Reply Generator │ (Short prompt generation matching Spotify tone)
└──────────────┬───────────────┘
               │ (Draft Reply)
               ▼
┌──────────────────────────────┐
│ Stage 4: Escalation Engine   │ (Deterministic business rule risk engine)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Stage 5: Structured JSON Output
└──────────────────────────────┘
```

---

## 4. Baseline Approaches & Intent Model Comparison

We evaluated three levels of models on the 200-sample Golden Evaluation Dataset:

1. **Baseline 1: Most Frequent Class Predictor (Majority Vote)**
   - Always predicts the most common intent category. Serves as the naive lower-bound benchmark ($1/8 = 12.5\%$).
2. **Baseline 2: TF-IDF + Logistic Regression**
   - Extracts unigram and bigram TF-IDF n-grams ($1000$ max features) and trains a linear Logistic Regression classifier.
3. **Final Model: Sentence Transformer Embeddings (`all-MiniLM-L6-v2`) + Logistic Regression**
   - Encodes customer queries into 384-dimensional dense semantic vectors using a pre-trained `all-MiniLM-L6-v2` transformer model, followed by a multi-class Logistic Regression classifier. **No deep neural network fine-tuning was performed.**

---

## 5. Evaluation Results

### Intent Classification Benchmark (200 Golden Samples)

| Model Approach | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Most Frequent** | 0.1250 | 0.0156 | 0.1250 | 0.0278 | 0.0278 |
| **Baseline 2: TF-IDF + LogReg** | 0.5750 | 0.6254 | 0.5750 | 0.5691 | 0.5691 |
| **Final: Sentence Transformer + LogReg** | **0.7550** | **0.7671** | **0.7550** | **0.7477** | **0.7477** |

### Reply Quality Evaluation (40-Sample Representative Subset)

Generated support replies were evaluated across 4 core dimensions (1–5 scale) comparing **LLM-as-a-Judge** scores with **Human Domain Expert** scores:

| Dimension | LLM Judge Avg | Human Evaluator Avg | Score Agreement ($\le 1$ pt) | Pearson Correlation |
| :--- | :---: | :---: | :---: | :---: |
| **Correctness** | 4.75 / 5.0 | 4.60 / 5.0 | **100.0%** | 0.798 |
| **Helpfulness** | 4.47 / 5.0 | 4.42 / 5.0 | **100.0%** | 0.552 |
| **Tone** | 5.00 / 5.0 | 4.83 / 5.0 | **100.0%** | N/A (Low Variance) |
| **Consistency** | 5.00 / 5.0 | 4.83 / 5.0 | **100.0%** | N/A (Low Variance) |

---

## 6. Failure Analysis

We extracted 5 real failure cases from empirical evaluation runs to analyze root causes and engineer future fixes:

1. **FAIL-001: Gift Card Activation Misclassified as Billing Dispute**
   - *Message*: "Redeemed a 12-month gift card but my account shows only 1 month active."
   - *Expected*: `Subscription & Premium` (Auto Handle) $ightarrow$ *Actual*: `Billing & Refund Request` (Escalate).
   - *Root Cause*: Overlapping financial tokens ('redeemed', '1 month active').
   - *Proposed Fix*: Add explicit gift voucher token disambiguation rules.

2. **FAIL-002: Playback Feature Issue Misclassified as App Crash**
   - *Message*: "Crossfade between tracks isn't working on my iOS app after latest update."
   - *Expected*: `Playback & Audio Bugs` $ightarrow$ *Actual*: `App Crash & Performance`.
   - *Root Cause*: Keyword 'after latest update' dominated feature weighting over 'Crossfade'.
   - *Proposed Fix*: Weight audio domain feature terms higher than generic update keywords.

3. **FAIL-003: Login Restriction Error Misclassified as Security Breach**
   - *Message*: "Multi-device login restriction error popup popping up unexpectedly."
   - *Expected*: `Login & Auth Issue` $ightarrow$ *Actual*: `Account Compromise & Security`.
   - *Root Cause*: Conservative security classifier associated remote device warnings with account hacking.
   - *Proposed Fix*: Pre-screen for technical error code strings before escalation.

4. **FAIL-004: Mid-Month Plan Drop Ambiguity**
   - *Message*: "Why did my Premium status drop back to Free mid-month?"
   - *Expected*: `Subscription & Premium` $ightarrow$ *Actual*: `Billing & Refund Request`.
   - *Root Cause*: Boundary blurring between payment sync steps and billing cancellation cycles.
   - *Proposed Fix*: Implement secondary confidence thresholding for dual-intent queries.

5. **FAIL-005: Feature Query Retrieval Context Mismatch**
   - *Message*: "Can I disable podcast recommendations from showing up on my home feed?"
   - *Expected*: `General Query & Feature Request` $ightarrow$ *Actual*: `App Crash & Performance`.
   - *Root Cause*: Retrieval index returned UI rendering crash cases.
   - *Proposed Fix*: Filter FAISS retrieval search space using predicted intent metadata.

---

## 7. Limitations

1. **Synthetic Corpus Expansion**: While historical Q&A seeds are extracted from real patterns, corpus expansion relies on structured template variations.
2. **Offline Local Vector Search**: FAISS `IndexFlatIP` operates in-memory without persistent cloud synchronization.
3. **Keyword-Dominant Escalation**: Rule-based escalation relies on intent classification accuracy; misclassifying a security query as a general query could bypass human review.

---

## 8. What Is Misleading About My Headline Number?

Our headline intent classification accuracy is **75.5%**. While impressive compared to the 12.5% majority baseline, presenting 75.5% as "overall system performance" is misleading for three reasons:

1. **Balanced Evaluation vs. Real-World Skew**: The Golden Dataset has equal 12.5% distribution per intent. In live production, `Login & Auth` and `Playback Bugs` account for >60% of volume, meaning live accuracy will differ significantly.
2. **Intent Accuracy $
eq$ Resolution Accuracy**: A message correctly classified as `Subscription & Premium` (75.5% accuracy) can still receive an unhelpful reply if FAISS retrieves an irrelevant case (e.g. FAIL-005).
3. **Escalation Safety Bias**: The 75.5% metric treats all errors equally. Misclassifying `Account Security` as `General Query` is far more dangerous than misclassifying `Playback Bugs` as `App Crash`.

---

## 9. What I Would Do With One More Week

If given one additional week, I would prioritize the following engineering enhancements:

1. **Hybrid Search (BM25 + FAISS Dense Retrieval)**: Combine sparse lexical keyword matching (BM25) with dense embeddings to prevent failure modes like FAIL-005.
2. **Confidence Thresholding & Clarification Routing**: If the classifier's top intent probability is $<0.55$, automatically prompt the user with a 2-option clarifying question rather than making a low-confidence decision.
3. **Intent-Filtered FAISS Indexing**: Partition the FAISS vector index by intent category so retrieval only queries historical cases within the predicted intent.
4. **FastAPI REST API & React Dashboard**: Package `src/pipeline.py` into a lightweight REST endpoint (`/api/v1/support/process`) with a React support agent triage dashboard.
