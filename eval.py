# eval.py
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph

graph = build_graph()

results = []
for fixture_file in Path("fixtures/processed").rglob("*.json"):
    fixture = json.loads(fixture_file.read_text())
    result = graph.invoke({
        "raw_excerpt": fixture["raw_excerpt"],
        "test_name":   fixture["test_name"],
        "runner":      fixture["runner"],
        "timestamp":   fixture["timestamp"],
        "repo":        fixture.get("repo","python/cpython"),
    })
    correct = result["failure_type"] == fixture["failure_type"]
    results.append(correct)
    status = "✓" if correct else "✗"
    print(f"{status} {fixture_file} → {result['failure_type']} (gt: {fixture['failure_type']})")

accuracy = sum(results) / len(results)
print(f"\nTriage accuracy: {accuracy:.0%} ({sum(results)}/{len(results)})")