import os
import json
import re
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from agent.state import AgentState


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_retries=3, 
    )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def get_test_run_history(repo: str, test_name: str, lookback_runs: int = 50) -> dict:
    """
    Retrieves recent CI run results for a specific test name by downloading
    logs for each run and searching for pass/fail of that test.

    Args:
        repo: GitHub repo in 'owner/name' format e.g. 'python/cpython'
        test_name: the name of the test to look up
        lookback_runs: how many recent completed runs to check
    """
    import zipfile
    import io

    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    owner, name = repo.split("/")

    # Step 1 — fetch recent completed runs
    runs_url = f"https://api.github.com/repos/{owner}/{name}/actions/runs"
    response = requests.get(
        runs_url,
        headers=headers,
        params={"per_page": lookback_runs, "status": "completed"}
    )
    response.raise_for_status()
    runs = response.json()["workflow_runs"]

    # Step 2 — for each run, download logs and search for the test name
    results = []
    for run in runs:
        run_id = run["id"]
        logs_url = f"https://api.github.com/repos/{owner}/{name}/actions/runs/{run_id}/logs"

        try:
            log_response = requests.get(
                logs_url,
                headers=headers,
                allow_redirects=True,
                timeout=10
            )
            if log_response.status_code != 200:
                continue

            # Unzip and search all log files for the test name
            zip_file = zipfile.ZipFile(io.BytesIO(log_response.content))
            run_outcome = "not_found"

            for log_name in zip_file.namelist():
                content = zip_file.open(log_name).read().decode("utf-8", errors="replace")

                # Check for explicit pass or fail of this specific test
                if f"{test_name} ... ok" in content or f"PASSED {test_name}" in content or f"{test_name} PASSED" in content:
                    run_outcome = "passed"
                    break
                elif f"{test_name} ... FAIL" in content or f"FAILED {test_name}" in content \
                        or f"{test_name} ... ERROR" in content or f"{test_name} FAILED" in content:
                    run_outcome = "failed"
                    break

            if run_outcome != "not_found":
                results.append({
                    "run_id": run_id,
                    "created_at": run["created_at"],
                    "outcome": run_outcome,
                })

        except Exception:
            # Skip runs where logs are unavailable or expired
            continue

    # Step 3 — compute flaky probability from test-level results
    total = len(results)
    failures = sum(1 for r in results if r["outcome"] == "failed")
    passes = sum(1 for r in results if r["outcome"] == "passed")
    flaky_probability = round(failures / total, 2) if total > 0 else 0.0

    return {
        "test_name": test_name,
        "runs_checked": lookback_runs,
        "runs_with_test_found": total,
        "failures": failures,
        "passes": passes,
        "flaky_probability": flaky_probability,
        "verdict": "flaky" if 0.1 < flaky_probability < 0.9 else "consistent",
        "history": results,  # full per-run breakdown for the LLM to reason over
    }

# ---------------------------------------------------------------------------
# Per-type prompt templates 
# ---------------------------------------------------------------------------

ANALYSIS_PROMPTS = {

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
You are analyzing an environment issue — the test failed because of a missing dependency,
wrong library version, missing binary, or infrastructure problem.

Your goal:
1. Identify the exact missing or misconfigured component
2. Determine the scope: affects only this test, this module, or the entire test suite
3. Provide the exact command or config change needed to fix it

Severity criteria — apply strictly:
- critical: import failure or missing binary that blocks entire test collection
- high: failure in test setup that blocks a specific module
- medium: failure that affects a single test, workaround exists
- low: non-blocking warning or informational issue

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

def _extract_text(response) -> str:
    """Safely extracts text content from an LLM response regardless of format."""
    if isinstance(response.content, str):
        return response.content
    # content is a list of blocks — find the text block
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block["text"]
        if hasattr(block, "type") and block.type == "text":
            return block.text
    return ""

# ---------------------------------------------------------------------------
# Flaky analysis node — with tool calling
# ---------------------------------------------------------------------------

def flaky_analysis_node(state: AgentState) -> AgentState:
    """
    Analyzes flaky test failures. Uses get_test_run_history tool when the
    excerpt alone does not contain clear evidence of non-determinism.
    """
    llm_with_tools = get_llm().bind_tools([get_test_run_history])

    prompt = f"""
You are analyzing a potentially flaky test failure.

You have access to a tool that retrieves recent CI run history for a repository.
Use it if the excerpt does not contain clear evidence of flakiness such as
explicit intermittent language, timing variance, or historical pass rate.

Test name: {state["test_name"]}
Repo: {state["repo"]}

Based on the excerpt and any tool results, determine:
1. Whether this is genuinely flaky or a consistent failure mislabelled as flaky
2. The estimated flaky probability (0.0 to 1.0)
3. The most likely non-determinism source: timing, concurrency, external service,
   random seed, resource leak, or order-dependent state
4. The recommended fix

Return JSON only:
{{
  "root_cause": "<non-determinism source or reason it is not actually flaky>",
  "recommended_action": "<concrete fix: retry logic, mock, isolation, or escalate to logic_bug>",
  "severity": "<critical|high|medium|low>",
  "flaky_probability": <0.0 to 1.0>,
  "tool_was_called": <true|false>
}}

Excerpt:
{state["raw_excerpt"]}
"""

    messages = [HumanMessage(content=prompt)]
    response = llm_with_tools.invoke(messages)

     # Loop until the model stops making tool calls
    max_iterations = 5  # safety cap to prevent infinite loops
    iterations = 0

    while response.tool_calls and iterations < max_iterations:
        iterations += 1
        tool_call = response.tool_calls[0]
        tool_result = get_test_run_history.invoke(tool_call["args"])

        messages.append(response)
        messages.append(
            ToolMessage(
                content=json.dumps(tool_result),
                tool_call_id=tool_call["id"],
            )
        )
        response = llm_with_tools.invoke(messages)

    text = _extract_text(response)
    if not text:
        raise ValueError(f"No text content in response: {response.content}")

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in flaky analysis response:\n{text}")

    result = json.loads(match.group())

    return {
        **state,
        "root_cause": result.get("root_cause", "Not determined"),
        "recommended_action": result.get("recommended_action", "Not determined"),
        "severity": result.get("severity", "medium"),
    }


# ---------------------------------------------------------------------------
# Generic analysis node — all other failure types
# ---------------------------------------------------------------------------

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