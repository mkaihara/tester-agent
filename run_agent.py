import os   
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not set — check your .env file")
print(f"API key loaded: {api_key[:12]}...")

import json
from agent.graph import build_graph

# Load one fixture
fixture_path = Path("fixtures/processed/logic_bug/001.json")
fixture = json.loads(fixture_path.read_text())

# Run the agent
graph = build_graph()
result = graph.invoke({
    "raw_excerpt": fixture["raw_excerpt"],
    "test_name": fixture["test_name"],
    "runner": fixture["runner"],
    "timestamp": fixture["timestamp"],
})

print(f"Ground truth:  {fixture['failure_type']}")
print(f"Agent output:  {result['failure_type']} ({result['triage_confidence']})")