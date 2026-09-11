"""
src/case_retrieval.py
---------------------
Lightweight local vector retrieval engine for SpotifyCares historical support cases.
Uses SentenceTransformers (all-MiniLM-L6-v2) for embedding generation
and FAISS (IndexFlatIP with L2 normalized vectors = Cosine Similarity) for fast similarity search.
"""

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CLEAN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spotify_cleaned_pairs.csv")

class CaseRetrievalEngine:
    """
    Local FAISS retrieval engine.
    Indexes historical Spotify customer support Q&A pairs and retrieves top-k similar cases.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', data_path=CLEAN_DATA_PATH):
        self.encoder = SentenceTransformer(model_name)
        self.data_path = data_path
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        messages = self.df["customer_message"].astype(str).tolist()

        # Encode messages into dense vectors
        embeddings = self.encoder.encode(messages, show_progress_bar=False, convert_to_numpy=True)

        # Normalize vectors for Cosine Similarity using Inner Product (IP) index
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"[CaseRetrievalEngine] Successfully indexed {self.index.ntotal} historical Spotify cases.")

    def retrieve_top_k(self, query: str, k: int = 3):
        """
        Retrieves top k historical Q&A cases given a customer message query.
        Returns a list of dicts with: customer_message, support_reply, intent, similarity_score.
        """
        query_vec = self.encoder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "case_id": row.get("pair_id", f"SPOT-{idx:04d}"),
                "intent": row.get("intent_category", "General Query & Feature Request"),
                "customer_message": row.get("customer_message", ""),
                "support_reply": row.get("support_reply", ""),
                "similarity_score": round(float(score), 4)
            })
        return results

if __name__ == "__main__":
    retriever = CaseRetrievalEngine()
    test_query = "My Spotify desktop app crashed when I opened my playlists"
    cases = retriever.retrieve_top_k(test_query, k=3)
    print(f"\nTest Query: {test_query}\nRetrieved Top 3 Cases:")
    for i, c in enumerate(cases, start=1):
        print(f"{i}. [{c['intent']}] (Score: {c['similarity_score']}) Q: {c['customer_message']} -> A: {c['support_reply']}")
