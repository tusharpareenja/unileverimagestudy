"""Domain helpers for tenant/partner-specific behaviour (e.g. Unilever)."""

# Reserved classification question for special creators (e.g. Unilever): "Do you like this fragrance?"
# Injected with order=0 so it appears first in participate flow and in flattened CSV (exported as 1/0).
# question_id is String(10) in DB so we use "Q0".
FRAGRANCE_QUESTION_ID = "Q0"
FRAGRANCE_QUESTION_TEXT = "Do you like this fragrance?"
FRAGRANCE_ANSWER_OPTIONS = [{"id": "Y", "text": "Yes"}, {"id": "N", "text": "No"}]
FRAGRANCE_ORDER = 0


def is_unilever_domain(email: str) -> bool:
    """Return True if the email's domain is unilever.com (case-insensitive)."""
    if not email or not isinstance(email, str):
        return False
    parts = email.strip().rsplit("@", 1)
    if len(parts) != 2:
        return False
    return parts[1].lower() == "unilever.com"
