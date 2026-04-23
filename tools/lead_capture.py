"""
lead_capture.py
Mock lead-capture tool that simulates a CRM API call.
Only called when the agent has collected name, email, and platform.
"""

import re
from datetime import datetime


def _validate_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"
    return bool(re.match(pattern, email))


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead.

    Args:
        name:     Full name of the prospect
        email:    Email address
        platform: Creator platform (YouTube, Instagram, etc.)

    Returns:
        dict with status and confirmation message.

    Raises:
        ValueError: If any required field is missing or email is invalid.
    """
    # Validate inputs
    if not name or not name.strip():
        raise ValueError("Lead name cannot be empty.")
    if not email or not _validate_email(email.strip()):
        raise ValueError(f"Invalid email address: {email}")
    if not platform or not platform.strip():
        raise ValueError("Creator platform cannot be empty.")

    name = name.strip()
    email = email.strip().lower()
    platform = platform.strip()

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Simulate the CRM call
    print(f"\n{'='*50}")
    print(f"  ✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {timestamp}")
    print(f"{'='*50}\n")

    return {
        "status": "success",
        "message": f"Lead captured successfully: {name}, {email}, {platform}",
        "lead_id": f"LEAD-{abs(hash(email)) % 100000:05d}",
        "timestamp": timestamp,
    }
