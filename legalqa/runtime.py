"""Optional cooperative limits; the notebook supervisor also enforces a hard timeout."""
import os
import time


def should_pause(completed=0):
    cap = int(os.environ.get("LEGALQA_MAX_ITEMS", "0"))
    deadline = float(os.environ.get("LEGALQA_DEADLINE", "0"))
    return bool((cap and completed >= cap) or
                (deadline and time.time() >= deadline - 180))
