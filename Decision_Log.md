# Decision Log: SpotifyCares AI Support Assistant

Documenting 12 core engineering and architectural decisions made during development.

---

### Decision 1: Focus on `@SpotifyCares` Dataset
- **Choice**: Selected `@SpotifyCares` as the target brand dataset.
- **Alternatives**: `@AppleSupport`, `@AmazonHelp`, `@Uber_Support`.
- **Reasoning**: Spotify support covers clear, distinct technical, financial, and account security issues. Its Twitter interactions follow consistent Q&A patterns without the logistical complexity of retail shipping or rideshare tracking.

---

### Decision 2: Complete Code & Evaluation Before Writing Docs
- **Choice**: Deferred writing `README.md`, `Mandatory_Report.md`, and `Decision_Log.md` until the pipeline and evaluation scripts were fully built and tested.
- **Alternatives**: Writing documentation concurrently during early development.
- **Reasoning**: Ensures all benchmark metrics, failure cases, and system trade-offs in the final report are backed by actual runtime data rather than initial assumptions.

---

### Decision 3: Use a 200-Sample Golden Evaluation Set for Intents
- **Choice**: Manually labeled 200 customer messages with intent tags for classification evaluation.
- **Alternatives**: Writing 200 custom reference replies by hand.
- **Reasoning**: 200 intent labels provide a solid, statistically reliable benchmark for model accuracy. Reply quality is better evaluated on a smaller 40-sample representative subset using human domain scoring and automated heuristics.

---

### Decision 4: SentenceTransformer (`all-MiniLM-L6-v2`) + Logistic Regression Classifier
- **Choice**: Combined pre-trained sentence embeddings with a multi-class Logistic Regression classifier.
- **Alternatives**: Fine-tuning BERT/RoBERTa models, Deep Neural Networks, XGBoost.
- **Reasoning**: Avoids heavy deep-learning fine-tuning, trains locally in under 2 seconds, runs reliably on any laptop, and achieves strong classification accuracy (75.5%).

---

### Decision 5: Local FAISS (`IndexFlatIP`) for Historical Case Retrieval
- **Choice**: Indexed sentence embeddings into an in-memory FAISS inner-product vector index.
- **Alternatives**: Managed vector databases like Pinecone, Chroma, or Milvus.
- **Reasoning**: Eliminates external API dependencies and cloud database setup, providing sub-millisecond local search times.

---

### Decision 6: Rule-Based Engine for Escalation Decisions
- **Choice**: Built a deterministic, rule-based decision engine to flag whether an issue is `Auto Handle` or `Escalate`.
- **Alternatives**: Training an ML model to predict escalation status.
- **Reasoning**: Safety-critical business logic (like handling security breaches or billing refunds) needs to be 100% predictable, auditable, and easy to update without retraining a model.

---

### Decision 7: Concise Prompt Design for Support Replies
- **Choice**: Used direct, minimal prompt structures (Query + Intent + Top-3 Similar Cases).
- **Alternatives**: Complex multi-agent setups or multi-step reasoning prompts.
- **Reasoning**: Keeps latency low, minimizes API dependency, and reliably generates short, friendly support replies matching Spotify’s Twitter tone.

---

### Decision 8: Fallback Template Synthesizer for Local Execution
- **Choice**: Built a local template-based fallback reply generator when LLM API keys are not provided.
- **Alternatives**: Requiring an API key to run the project or letting the script fail.
- **Reasoning**: Guarantees that any interviewer or reviewer can clone the repository and test the end-to-end pipeline 100% offline without needing paid API keys.

---

### Decision 9: 4-Metric Framework for Reply Evaluation
- **Choice**: Evaluated generated replies across 4 key dimensions (Correctness, Helpfulness, Tone, Consistency) comparing automated LLM judge scores against manual ratings.
- **Alternatives**: Traditional n-gram overlap metrics like BLEU or ROUGE.
- **Reasoning**: BLEU and ROUGE heavily penalize valid, rephrased support replies. Evaluating on core qualities like tone and correctness aligns much better with real-world customer support standards.

---

### Decision 10: Taxonomy of 8 Mutually Exclusive Business Intents
- **Choice**: Structured intent classification around 8 core categories (Auth, Subscription, Billing, Audio Bugs, App Performance, Security, Playlists, and General Queries).
- **Alternatives**: 3 broad categories or 25+ fine-grained sub-intents.
- **Reasoning**: 8 categories cover over 95% of typical support queries while keeping class boundaries clear enough for efficient ML classification.

---

### Decision 11: L2-Normalized Inner Product (IP) FAISS Index
- **Choice**: Applied `faiss.normalize_L2` to feature vectors before inserting them into `IndexFlatIP`.
- **Alternatives**: Raw Euclidean distance (`IndexFlatL2`).
- **Reasoning**: Normalizing vector lengths makes the Inner Product mathematically identical to Cosine Similarity, which works best for semantic text matching.

---

### Decision 12: Standardized Single JSON Output Structure
- **Choice**: Formatted final output as a clean JSON object containing `{intent, reply, decision, reason}`.
- **Alternatives**: Free-form text output or nested complex metadata.
- **Reasoning**: Matches the target format specified in the problem statement and makes the output immediately ready for downstream API integration.
