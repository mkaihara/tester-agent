import os
import json
import re
from langchain_anthropic import ChatAnthropic
from agent.state import AgentState


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )


# ---------------------------------------------------------------------------
# Per-type prompt templates
# Each one looks for different signals and recommends different actions
# ---------------------------------------------------------------------------

ANALYSIS_PROMPTS = {

    "flaky": """
You are analyzing a flaky test failure — a test that passes sometimes and fails sometimes.

Your goal:
1. Identify the non-determinism source: timing, concurrency, external service, random seed, resource leak, or order-dependent state
2. Estimate how often this test likely fails (if inferable)
3. Recommend a concrete fix: retry logic, test isolation, mock the external dependency, fix the race condition

Return JSON only:
{{
  "root_cause": "<one sentence identifying the specific non-determinism source>",
  "recommended_action": "<concrete fix, not generic advice>",
  "severity": "<critical|high|medium|low>",
  "rerun_suggested": <true|false>
}}

Test name: {test_name}
Excerpt:
{raw_excerpt}
""",

    "regression": """
You are analyzing a regression — a test that was passing before and is now failing.

Your goal:
1. Identify what value or behavior changed — be specific about expected vs actual
2. Hypothesize what kind of code change could have caused this: API change, dependency update, config change, data format change
3. Recommend the immediate action: revert, fix forward, or add a compatibility shim

Return JSON only:
{{
  "root_cause": "<one sentence describing what changed and from what to what>",
  "recommended_action": "<revert or fix-forward recommendation with rationale>",
  "severity": "<critical|high|medium|low>",
  "bisect_suggested": <true|false>
}}

Test name: {test_name}
Excerpt:
{raw_excerpt}
""",

    "env_issue": """
You are analyzing an environment issue — the test failed because of a missing dependency, wrong library version, missing binary, or infrastructure problem.

Your goal:
1. Identify the exact missing or misconfigured component
2. Determine the scope: affects only this test, this module, or the entire test suite
3. Provide the exact command or config change needed to fix it

Severity criteria:
- critical: blocks entire test suite collection or CI pipeline
- high: blocks a specific test module
- medium: affects a single test, workaround exists
- low: informational, non-blocking

Return JSON only:
{{
  "root_cause": "<the specific missing or misconfigured component>",
  "recommended_action": "<exact install command or config fix>",
  "severity": "<critical|high|medium|low>",
  "affects_suite": <true|false>
}}

Test name: {test_name}
Excerpt:
{raw_excerpt}
""",

    "logic_bug": """
You are analyzing a logic bug — a test that consistently fails because the code produces the wrong output.

Your goal:
1. Identify the exact assertion that failed and what the discrepancy is
2. Hypothesize the likely location of the bug: off-by-one, wrong formula, incorrect boundary condition, unhandled edge case
3. Suggest the minimal investigation step: which function to inspect first

Return JSON only:
{{
  "root_cause": "<the specific assertion failure and likely bug category>",
  "recommended_action": "<which function or module to inspect and why>",
  "severity": "<critical|high|medium|low>",
  "deterministic": true
}}

Test name: {test_name}
Excerpt:
{raw_excerpt}
""",

    "timeout": """
You are analyzing a timeout failure — a test that exceeded its time limit and was killed.

Your goal:
1. Determine the timeout type: runner wall time, in-code timeout, or network read timeout
2. Identify whether this is a performance regression, an infinite loop, or an external service hanging
3. Recommend whether to increase the timeout, fix the performance issue, or mock the slow dependency

Return JSON only:
{{
  "root_cause": "<timeout type and likely cause>",
  "recommended_action": "<increase timeout, fix performance, or mock — with rationale>",
  "severity": "<critical|high|medium|low>",
  "infrastructure_issue": <true|false>
}}

Test name: {test_name}
Excerpt:
{raw_excerpt}
""",
}


def analysis_node(state: AgentState) -> AgentState:
    failure_type = state["failure_type"]

    if failure_type not in ANALYSIS_PROMPTS:
        return {
            **state,
            "root_cause": "Unknown failure type — no analysis performed.",
            "recommended_action": "Manual inspection required.",
            "severity": "high",
        }

    prompt = ANALYSIS_PROMPTS[failure_type].format(
        test_name=state["test_name"],
        raw_excerpt=state["raw_excerpt"],
    )

    llm = get_llm()
    response = llm.invoke(prompt)

    text = response.content
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in analysis response:\n{text}")

    result = json.loads(match.group())

    return {
        **state,
        "root_cause": result.get("root_cause", "Not determined"),
        "recommended_action": result.get("recommended_action", "Not determined"),
        "severity": result.get("severity", "medium"),
    }