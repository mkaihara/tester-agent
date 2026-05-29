# explain_fixtures.py
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(Path(__file__).parent / ".env")

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

fixtures_to_explain = list(Path("fixtures/processed").rglob("*.json"))

for fixture_path in sorted(fixtures_to_explain):
    fixture = json.loads(fixture_path.read_text())

    prompt = f"""
You are a senior software engineer reviewing a CI test failure.

Read the following test output excerpt and explain in plain English:
1. What happened — describe the failure in one sentence
2. What type of failure this is, choosing from:
   - flaky: passes sometimes, fails sometimes
   - regression: was working before, now broken
   - env_issue: missing dependency, wrong environment, import error
   - logic_bug: consistent wrong output or assertion mismatch
   - timeout: exceeded time limit
3. Why you chose that type — cite specific words or patterns from the excerpt
4. Confidence: high / medium / low and why

Excerpt:
{fixture["raw_excerpt"]}
"""
    response = llm.invoke(prompt)

    print(f"\n{'='*60}")
    print(f"FILE:     {fixture_path}")
    print(f"REGEX LABEL (may be wrong): {fixture['failure_type']}")
    print(f"\nLLM EXPLANATION:\n{response.content}")
    print(f"{'='*60}")