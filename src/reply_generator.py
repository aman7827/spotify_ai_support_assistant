"""
src/reply_generator.py
----------------------
Reply Generation engine for SpotifyCares AI Support Assistant.
Constructs short, concise prompts using predicted intent and top 3 retrieved cases.
Supports OpenAI API (gpt-4o-mini / gpt-3.5-turbo) or fallback local generator.
"""

import os

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class ReplyGenerator:
    """
    Generates realistic Spotify Support replies based on customer query,
    predicted intent, and retrieved historical resolution pairs.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

    def generate_reply(
        self,
        customer_message,
        predicted_intent,
        retrieved_cases,
    ):
        """Generate a concise reply using OpenAI when configured, or a fallback."""
        retrieved_cases = retrieved_cases or []

        if HAS_OPENAI and self.api_key:
            try:
                self.client = self.client or OpenAI(api_key=self.api_key)
                context_str = "\n".join(
                    case.get("support_reply", "")
                    for case in retrieved_cases[:3]
                    if case.get("support_reply")
                ) or "No similar cases available."
                prompt = f"""
Customer Message: "{customer_message}"
Intent: "{predicted_intent}"

Similar Historical Cases:
{context_str}

Instruction: Write a short, helpful Spotify support reply."""
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.4,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(
                    f"[ReplyGenerator] OpenAI call failed ({e}); using template fallback."
                )

        if retrieved_cases:
            top_case = retrieved_cases[0]
            top_reply = top_case.get("support_reply", "")
            if top_reply:
                return f"Hi! {top_reply}"

        intent = str(predicted_intent).lower()
        return (
            f"Hey! We'd be glad to help with your {intent}. "
            "Send us a DM with your account details so we can investigate!"
        )
