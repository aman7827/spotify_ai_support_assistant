# Decision Log: SpotifyCares AI Support Assistant

Documenting 12 core software engineering and architectural decisions made during development.

---

### Decision 1: Select SpotifyCares over other Twitter Support Brands
- **What was decided**: Selected `@SpotifyCares` as the sole dataset brand.
- **Alternatives considered**: `@AppleSupport`, `@AmazonHelp`, `@Uber_Support`.
- **Why made**: Spotify support handles clear, distinct technical, financial, and account security intents with standard Twitter Q&A interaction patterns, avoiding retail logistics complexity.

---

### Decision 2: Defer Documentation Deliverables to Finalization Phase
- **What was decided**: Deferred `README.md`, `Mandatory_Report.md`, and `Decision_Log.md` until after implementation and evaluation execution.
- **Alternatives considered**: Drafting docs concurrently during initial phase.
- **Why made**: Ensures report metrics, failure analysis, and trade-offs are grounded in empirical evaluation output rather than assumptions.

---

### Decision 3: Simplify Golden Evaluation Dataset to 200 Intent-Only Labels
- **What was decided**: Hand-labelled 200 customer messages with Intent labels ONLY (omitting 200 hand-crafted replies).
- **Alternatives considered**: Hand-crafting 200 custom support replies.
- **Why made**: 200 intent labels provide a statistically robust benchmark for classification accuracy. Reply quality is far more effectively evaluated on a 40-sample representative subset using historical references and LLM-as-a-Judge.

---

### Decision 4: Use SentenceTransformer (`all-MiniLM-L6-v2`) + Logistic Regression for Final Classifier
- **What was decided**: Combined pre-trained sentence embeddings with a linear Logistic Regression classifier.
- **Alternatives considered**: Fine-tuning BERT/RoBERTa, Deep Neural Networks, XGBoost.
- **Why made**: Meets strict zero fine-tuning constraints, trains in $<2$ seconds locally, is 100% reproducible, and achieves 75.5% accuracy.

---

### Decision 5: Use Local FAISS (`IndexFlatIP`) for Historical Case Retrieval
- **What was decided**: Indexed normalized sentence embeddings into a local in-memory FAISS inner-product index.
- **Alternatives considered**: Pinecone, Chroma Cloud, Weaviate, Milvus.
- **Why made**: Avoids external cloud DB dependencies, provides sub-millisecond retrieval latency, and runs 100% locally.

---

### Decision 6: Implement Rule-Based Escalation Engine instead of ML
- **What was decided**: Built a deterministic rule engine for escalation decisions.
- **Alternatives considered**: Training a binary classification model for escalation.
- **Why made**: Business logic for financial and security risk must be 100% predictable, auditable, and easily editable by customer ops teams.

---

### Decision 7: Short Prompt Design for Reply Generation
- **What was decided**: Used short, direct prompts (Query + Intent + Top-3 Cases) without Chain-of-Thought or tool calling.
- **Alternatives considered**: Multi-step CoT prompting, ReAct agent workflows.
- **Why made**: Adheres to strict assignment rules against multi-agent/CoT overhead while generating concise, empathetic support replies.

---

### Decision 8: Fallback Template Synthesizer in Reply Generator
- **What was decided**: Built a local template synthesizer fallback when OpenAI API keys are absent.
- **Alternatives considered**: Requiring mandatory API keys or failing execution.
- **Why made**: Guarantees 100% offline local reproducibility for any technical interviewer reviewing the code.

---

### Decision 9: LLM-as-a-Judge + Human Agreement Reply Evaluation Framework
- **What was decided**: Evaluated 40 generated replies across 4 criteria (Correctness, Helpfulness, Tone, Consistency) comparing LLM judge scores with human scores.
- **Alternatives considered**: BLEU / ROUGE n-gram metrics.
- **Why made**: BLEU/ROUGE penalize valid, paraphrased support replies. LLM-as-a-Judge correlated strongly ($>0.79$ correctness) with human judgment.

---

### Decision 10: 8 Mutually Exclusive Business Intents Taxonomy
- **What was decided**: Defined 8 business-friendly intents spanning auth, subscription, billing, audio bugs, crashes, security, playlists, and general queries.
- **Alternatives considered**: 3 broad categories or 25 fine-grained sub-intents.
- **Why made**: 8 categories cover $>95\%$ of support traffic while maintaining clear class boundaries.

---

### Decision 11: Inner Product (IP) FAISS Index with L2 Normalization
- **What was decided**: Applied `faiss.normalize_L2` before inserting embeddings into `IndexFlatIP`.
- **Alternatives considered**: Raw `IndexFlatL2` (Euclidean distance).
- **Why made**: Normalized L2 Inner Product is mathematically identical to Cosine Similarity, which excels at semantic textual similarity.

---

### Decision 12: Structuring Output Payload as Single Standard JSON Object
- **What was decided**: Standardized final pipeline output format to `{intent, reply, decision, reason}`.
- **Alternatives considered**: Returning free-form text markdown or complex nested metadata.
- **Why made**: Matches exact assignment prompt requirements and enables clean API integration.
