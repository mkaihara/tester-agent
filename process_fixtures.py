import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


# ---------------------------------------------------------------------------
# LLM-based classification — replaces regex heuristics
# ---------------------------------------------------------------------------

def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )


def classify_failure(excerpt: str) -> str:
    """
    Classifies a test failure excerpt using an LLM.
    Falls back to 'logic_bug' if classification fails.
    """
    llm = get_llm()

    prompt = f"""
Classify this CI test failure excerpt into exactly one of these types:

- timeout: exceeded time limit, process killed, wall time exceeded, ReadTimeout
- env_issue: missing dependency, ImportError, ModuleNotFoundError, binary not found, ConnectionRefused
- flaky: passes sometimes fails sometimes — requires explicit evidence of non-determinism such as ResourceWarning, random seed, timing variance, or historical pass rate
- regression: was passing before, now fails — requires explicit evidence of a prior passing state such as "previously", "used to", "before commit"
- logic_bug: consistent deterministic failure, assertion mismatch, wrong computed value — use this when no other type clearly fits

Disambiguation rules:
- flaky requires non-determinism evidence — a single failure with no such evidence is logic_bug
- regression requires prior passing state evidence — without it, use logic_bug
- env_issue takes priority over logic_bug when an ImportError or connection error is present
- timeout takes priority over all others when wall time or process kill is present

Return JSON only, no explanation:
{{"failure_type": "<one of the five types>", "confidence": "<high|medium|low>"}}

Excerpt:
{excerpt}
"""

    try:
        response = llm.invoke(prompt)
        text = response.content if isinstance(response.content, str) else ""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return "logic_bug"
        result = json.loads(match.group())
        failure_type = result.get("failure_type", "logic_bug")
        confidence = result.get("confidence", "low")
        print(f"    classified as: {failure_type} ({confidence})")
        return failure_type
    except Exception as e:
        print(f"    LLM classification failed: {e} — falling back to logic_bug")
        return "logic_bug"


# ---------------------------------------------------------------------------
# Extraction helpers (unchanged)
# ---------------------------------------------------------------------------

def extract_failure_section(log: str, context_lines: int = 80) -> str | None:
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
    patterns = [
        r"FAILED ([\w/\.]+::[\w]+)",
        r"ERROR: (test_\w+)",
        r"(test_\w+) \.\.\. (FAIL|ERROR)",
        r"File \".*\", line \d+, in (test_\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt)
        if match:
            return match.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Main processing loop (unchanged except for low-confidence warning)
# ---------------------------------------------------------------------------

def process_raw_logs(raw_dir: str = "fixtures/raw", output_dir: str = "fixtures/processed"):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    MAX_PER_TYPE = 20
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
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        repo = metadata.get("repo", "python/cpython")

        for log_file in sorted(run_dir.glob("*.txt")):
            content = log_file.read_text(encoding="utf-8", errors="replace")

            excerpt = extract_failure_section(content)
            if excerpt is None:
                continue

            print(f"  processing: {log_file.name}")
            failure_type = classify_failure(excerpt)

            if type_counters[failure_type] >= MAX_PER_TYPE:
                continue

            type_counters[failure_type] += 1
            idx = type_counters[failure_type]

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
            print(f"  [{failure_type}] {idx:03d} ← run {run_id} / {log_file.name}")

    print("\n--- Summary ---")
    for t, count in type_counters.items():
        status = "✓" if count >= MAX_PER_TYPE else f"⚠ only {count}"
        print(f"  {t}: {status}")


if __name__ == "__main__":
    process_raw_logs()