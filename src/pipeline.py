"""
src/pipeline.py
---------------
End-to-End Coordinator for SpotifyCares AI Support Assistant.
Connects:
Customer Message -> Intent Classifier -> Historical Case Retrieval -> Reply Generation -> Escalation Engine -> Structured JSON Output
"""

import os
import json
import argparse
import pandas as pd
from src.intent_classifier import SentenceTransformerClassifier
from src.case_retrieval import CaseRetrievalEngine
from src.reply_generator import ReplyGenerator
from src.escalation_engine import EscalationEngine
from src.data_prep import CLEAN_DATA_PATH

class SupportAssistantPipeline:
    """
    Main orchestration class that takes a raw customer message and outputs
    structured JSON matching exact assignment requirements:
    {
      "intent": "...",
      "reply": "...",
      "decision": "Auto Handle | Escalate",
      "reason": "..."
    }
    """
    def __init__(self):
        print("[Pipeline] Initializing SpotifyCares AI Support Assistant Pipeline...")
        self.retriever = CaseRetrievalEngine()
        self.classifier = SentenceTransformerClassifier()
        self.reply_gen = ReplyGenerator()
        self.escalation_engine = EscalationEngine()
        self._train_classifier()

    def _train_classifier(self):
        df = pd.read_csv(CLEAN_DATA_PATH)
        X = df["customer_message"].tolist()
        y = df["intent_category"].tolist()
        self.classifier.fit(X, y)
        print("[Pipeline] Trained Intent Classifier on historical corpus.")

    def process_message(self, customer_message: str) -> dict:
        intent = self.classifier.predict([customer_message])[0]

        cases = self.retriever.retrieve_top_k(customer_message, k=3)

        reply = self.reply_gen.generate_reply(customer_message, intent, cases)

        esc_result = self.escalation_engine.evaluate(customer_message, intent)

        return {
            "intent": intent,
            "reply": reply,
            "decision": esc_result["decision"],
            "reason": esc_result["reason"]
        }


def main():
    parser = argparse.ArgumentParser(description="SpotifyCares AI Support Assistant CLI")
    parser.add_argument("--message", type=str, help="Customer message to process")
    parser.add_argument("--test", action="store_true", help="Run automated test queries")
    args = parser.parse_args()

    pipeline = SupportAssistantPipeline()

    if args.message:
        res = pipeline.process_message(args.message)
        print("\n" + json.dumps(res, indent=2))
    elif args.test or not args.message:
        test_messages = [
            "I can't log into my Spotify account on my iPhone, keeps saying invalid password.",
            "You guys charged me twice $16.99 this month! I want my money back immediately.",
            "Songs keep pausing every 10 seconds on Android when screen locks.",
            "Someone hacked my account and changed my email address without permission!"
        ]
        print("\n================ RUNNING PIPELINE AUTOMATED TEST SUITE ================")
        for msg in test_messages:
            print(f"\n--- Customer Query: \"{msg}\" ---")
            res = pipeline.process_message(msg)
            print(json.dumps(res, indent=2))
        print("=======================================================================\n")

if __name__ == "__main__":
    main()
