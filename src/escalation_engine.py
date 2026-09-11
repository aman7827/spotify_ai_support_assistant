"""
src/escalation_engine.py
------------------------
Deterministic, Rule-Based Escalation Decision Engine for SpotifyCares.
Evaluates risk based on business intent & sentiment to output:
Decision ("Auto Handle" | "Escalate") + Reason.

DO NOT use ML for escalation to ensure 100% transparency and auditability.
"""

class EscalationEngine:
    """
    Evaluates customer intent and message rules to decide Auto Handle vs Escalate.
    """
    ESCALATION_RULES = {
        "Billing & Refund Request": {
            "decision": "Escalate",
            "reason": "Financial transaction / double billing / refund request requires secure account lookup and manual agent verification."
        },
        "Account Compromise & Security": {
            "decision": "Escalate",
            "reason": "Security alert / unauthorized email modification / account breach requires human verification to prevent identity fraud."
        }
    }
    
    AUTO_HANDLE_RULES = {
        "Login & Auth Issue": {
            "decision": "Auto Handle",
            "reason": "Standard authentication / password reset inquiry resolved via standard self-service troubleshooting steps."
        },
        "Subscription & Premium": {
            "decision": "Auto Handle",
            "reason": "General subscription plan status inquiry handled via standard account synchronization steps."
        },
        "Playback & Audio Bugs": {
            "decision": "Auto Handle",
            "reason": "Technical audio / playback issue handled via app settings and cache clearing instructions."
        },
        "App Crash & Performance": {
            "decision": "Auto Handle",
            "reason": "App stability / freezing issue handled via clean reinstallation and storage cache clearing guidelines."
        },
        "Playlist & Library Issues": {
            "decision": "Auto Handle",
            "reason": "Playlist management / local files sync query resolved via account library recovery steps."
        },
        "General Query & Feature Request": {
            "decision": "Auto Handle",
            "reason": "General feature inquiry / regional availability question handled via standard customer information reply."
        }
    }

    def evaluate(self, customer_message: str, predicted_intent: str) -> dict:
        if predicted_intent in self.ESCALATION_RULES:
            return self.ESCALATION_RULES[predicted_intent]

        message_lower = customer_message.lower()
        high_risk_keywords = ["lawsuit", "legal action", "fraud", "unauthorized charge", "hacked", "stolen card"]
        for kw in high_risk_keywords:
            if kw in message_lower:
                return {
                    "decision": "Escalate",
                    "reason": f"Customer message contains high-risk keyword ('{kw}') triggering mandatory human agent escalation."
                }

        if predicted_intent in self.AUTO_HANDLE_RULES:
            return self.AUTO_HANDLE_RULES[predicted_intent]

        return {
            "decision": "Auto Handle",
            "reason": "Standard inquiry auto-handled via general customer support guidelines."
        }
