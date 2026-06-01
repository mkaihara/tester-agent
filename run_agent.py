from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

import json
from agent.graph import build_graph

fixture_path = Path("fixtures/processed/flaky/001.json")
fixture = json.loads(fixture_path.read_text())

graph = build_graph()

result = graph.invoke(
    {
        "raw_excerpt": fixture["raw_excerpt"],
        "test_name":   fixture["test_name"],
        "runner":      fixture["runner"],
        "timestamp":   fixture["timestamp"],
        "repo":        fixture.get("repo", "python/cpython"),
        "eval_iterations":   0,      # ← new
        "evaluator_feedback": None,  # ← new
        "triage_reasoning":   None,  # ← new
    },
    config={
        "run_name": f"eval-{fixture['failure_type']}-{fixture_path.stem}",
    }   
)

print(result["report"])