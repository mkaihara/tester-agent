from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent / ".env")

import json
from agent.graph import build_graph

fixture_path = Path("fixtures/processed/env_issue/001_synthetic.json")
fixture = json.loads(fixture_path.read_text())

graph = build_graph()
result = graph.invoke({
    "raw_excerpt": fixture["raw_excerpt"],
    "test_name":   fixture["test_name"],
    "runner":      fixture["runner"],
    "timestamp":   fixture["timestamp"],
})

print(result["report"])