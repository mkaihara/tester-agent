# inspect_failures.py
import json
from pathlib import Path

misclassified = [
    "fixtures/processed/logic_bug/001.json",
    "fixtures/processed/logic_bug/003.json",
    "fixtures/processed/logic_bug/002.json",
    "fixtures/processed/timeout/003.json",
    "fixtures/processed/timeout/002.json",
    "fixtures/processed/env_issue/001.json",
    "fixtures/processed/env_issue/002.json",
    "fixtures/processed/regression/002.json",
    "fixtures/processed/regression/003.json",
]

for path in misclassified:
    fixture = json.loads(Path(path).read_text())
    print(f"\n{'='*60}")
    print(f"FILE:       {path}")
    print(f"GT LABEL:   {fixture['failure_type']}")
    print(f"TEST NAME:  {fixture['test_name']}")
    print(f"EXCERPT:\n{fixture['raw_excerpt']}")
    print(f"{'='*60}")