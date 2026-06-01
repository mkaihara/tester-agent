# eval.py
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph

graph = build_graph()

results = []
for fixture_file in Path("fixtures/processed").rglob("*.json"):
    fixture = json.loads(fixture_file.read_text())
    try:
        result = graph.invoke({
            "raw_excerpt":        fixture["raw_excerpt"],
            "test_name":          fixture["test_name"],
            "runner":             fixture["runner"],
            "timestamp":          fixture["timestamp"],
            "repo":               fixture.get("repo", "python/cpython"),
            "eval_iterations":    0,
            "evaluator_feedback": None,
            "triage_reasoning":   None,
        })
        correct = result["failure_type"] == fixture["failure_type"]
        results.append(correct)
        status = "✓" if correct else "✗"
        print(f"{status} {fixture_file} → {result['failure_type']} (gt: {fixture['failure_type']})")
    except Exception as e:
        print(f"⚠ {fixture_file} → ERROR: {e}")
        results.append(False)
    time.sleep(1)  # avoid rate limits

accuracy = sum(results) / len(results) if results else 0
print(f"\nTriage accuracy: {accuracy:.0%} ({sum(results)}/{len(results)})")