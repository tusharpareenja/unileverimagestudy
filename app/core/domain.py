"""Domain helpers for tenant/partner-specific behaviour (e.g. Unilever)."""


def is_unilever_domain(email: str) -> bool:
    """Return True if the email's domain is unilever.com (case-insensitive)."""
    if not email or not isinstance(email, str):
        return False
    parts = email.strip().rsplit("@", 1)
    if len(parts) != 2:
        return False
    return parts[1].lower() == "unilever.com"
