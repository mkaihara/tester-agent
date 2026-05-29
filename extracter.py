def extract_failure_section(log: str, context_lines: int = 50) -> str:
    """
    Finds the first ERROR or FAILED marker and returns
    surrounding context. Good enough for fixture extraction.
    """
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if any(marker in line for marker in ["FAILED", "ERROR", "ERROR:", "AssertionError"]):
            start = max(0, i - 10)
            end = min(len(lines), i + context_lines)
            return "\n".join(lines[start:end])
    return log[-3000:]  # fallback: last 3000 chars