import json
import re
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Classification heuristics
# Each entry is (failure_type, list_of_signals).
# Order matters — first match wins.
# ---------------------------------------------------------------------------

FAILURE_SIGNATURES = [
    ("timeout", [
        "timed out", "timeout", "exceeded", "killed", "wall time",
        "DurationWarning", "TIMEOUT"
    ]),
    ("env_issue", [
        "ModuleNotFoundError", "ImportError", "No such file",
        "Permission denied", "command not found", "library not found",
        "Segmentation fault", "core dumped", "OSError", "FileNotFoundError",
        "ConnectionRefusedError", "cannot open shared object"
    ]),
    ("flaky", [
        "intermittent", "flaky", "FLAKE", "ResourceWarning",
        "random seed", "non-deterministic", "occasionally fails"
    ]),
    ("regression", [
        "was", "previously", "expected.*but got", "used to",
        "broken since", "regressed", "AssertionError.*!=",
    ]),
    ("logic_bug", [
        "AssertionError", "assert ", "FAILED", "ERROR",
        "wrong result", "incorrect", "unexpected value"
    ]),
]


def classify_failure(text: str) -> str:
    text_lower = text.lower()
    for failure_type, signals in FAILURE_SIGNATURES:
        for signal in signals:
            if re.search(signal, text, re.IGNORECASE):
                return failure_type
    return "logic_bug"  # fallback — most generic


def extract_failure_section(log: str, context_lines: int = 60) -> str | None:
    """
    Finds the first hard failure marker and returns surrounding context.
    Returns None if no failure found (log is clean or non-test step).
    """
    failure_markers = [
        r"^(FAILED|ERROR|ERRORS)\b",
        r"= FAILURES =",
        r"= ERRORS =",
        r"Traceback \(most recent call last\)",
        r"AssertionError",
        r"test_\w+ \.\.\. (FAIL|ERROR)",
    ]
    
    lines = log.splitlines()
    for i, line in enumerate(lines):
        for marker in failure_markers:
            if re.search(marker, line):
                start = max(0, i - 10)
                end = min(len(lines), i + context_lines)
                return "\n".join(lines[start:end])
    
    return None


def extract_test_name(excerpt: str) -> str:
    """Best-effort extraction of the failing test name."""
    patterns = [
        r"FAILED ([\w/\.]+::[\w]+)",          # pytest style
        r"ERROR: (test_\w+)",                   # unittest style
        r"(test_\w+) \.\.\. (FAIL|ERROR)",     # verbose unittest
        r"File \".*\", line \d+, in (test_\w+)", # traceback
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt)
        if match:
            return match.group(1)
    return "unknown"


def process_raw_logs(raw_dir: str = "fixtures/raw", output_dir: str = "fixtures/processed"):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # Counters per failure type — we want max 3 per type
    MAX_PER_TYPE = 3
    type_counters: dict[str, int] = {
        "flaky": 0, "regression": 0,
        "env_issue": 0, "logic_bug": 0, "timeout": 0
    }

    for run_dir in sorted(raw_path.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            continue

        metadata = json.loads(metadata_file.read_text())
        timestamp = datetime.utcnow().isoformat() + "Z"  
        repo = metadata.get("repo", "python/cpython")

        for log_file in sorted(run_dir.glob("*.txt")):
            content = log_file.read_text(encoding="utf-8", errors="replace")

            excerpt = extract_failure_section(content)
            if excerpt is None:
                continue  # skip non-failure steps (setup, checkout, etc.)

            failure_type = classify_failure(excerpt)

            if type_counters[failure_type] >= MAX_PER_TYPE:
                continue  # already have enough of this type

            type_counters[failure_type] += 1
            idx = type_counters[failure_type]

            # Write fixture
            type_dir = output_path / failure_type
            type_dir.mkdir(parents=True, exist_ok=True)

            fixture = {
                "failure_type": failure_type,
                "repo": repo,
                "runner": "github-actions",
                "timestamp": timestamp,
                "test_name": extract_test_name(excerpt),
                "source_run_id": int(run_id),
                "source_file": log_file.name,
                "raw_excerpt": excerpt,
            }

            out_file = type_dir / f"{idx:03d}.json"
            out_file.write_text(json.dumps(fixture, indent=2))
            print(f"[{failure_type}] {idx:03d} ← run {run_id} / {log_file.name}")

    print("\n--- Summary ---")
    for t, count in type_counters.items():
        status = "✓" if count >= MAX_PER_TYPE else f"⚠ only {count}"
        print(f"  {t}: {status}")


if __name__ == "__main__":
    process_raw_logs()