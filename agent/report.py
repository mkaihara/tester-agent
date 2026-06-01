import os
from langchain_anthropic import ChatAnthropic
from agent.state import AgentState


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_retries=3, 
    )


def report_node(state: AgentState) -> AgentState:
    # Short-circuit for unclassified failures
    if state.get("failure_type") == "unclassified":
        report = (
            "CI FAILURE REPORT\n"
            "=================\n"
            f"Test:       {state['test_name']}\n"
            f"Runner:     {state['runner']}\n"
            f"Time:       {state['timestamp']}\n"
            f"Repo:       {state['repo']}\n\n"
            "CLASSIFICATION\n"
            "--------------\n"
            "Type:       unclassified\n"
            f"Confidence: low\n"
            f"Iterations: {state.get('eval_iterations', 0)}\n\n"
            "ROOT CAUSE\n"
            "----------\n"
            "The triage system and evaluator could not agree on a classification "
            f"after {state.get('eval_iterations', 0)} iterations. "
            "The excerpt may be ambiguous, truncated, or contain signals consistent "
            "with multiple failure types. Manual inspection is required.\n\n"
            "RECOMMENDED ACTION\n"
            "------------------\n"
            "Review the raw excerpt below and classify manually. Consider retrieving "
            "additional log context from the source CI run.\n\n"
            f"Raw excerpt:\n{state['raw_excerpt']}"
        )
        return {**state, "report": report}

    prompt = f"""
You are a technical writer producing a concise CI failure report for a software engineering team.

Write a structured plain-text report based on the following analysis. 
Be direct and actionable. No filler sentences. No markdown.

Input:
- Test name: {state["test_name"]}
- Runner: {state["runner"]}
- Timestamp: {state["timestamp"]}
- Failure type: {state["failure_type"]}
- Triage confidence: {state["triage_confidence"]}
- Severity: {state["severity"]}
- Root cause: {state["root_cause"]}
- Recommended action: {state["recommended_action"]}
- Repo: {state["repo"]}

Produce the report in this exact format:

CI FAILURE REPORT
=================
Test:       <test_name>
Runner:     <runner>
Time:       <timestamp>
Repo:       <repo>

CLASSIFICATION
--------------
Type:       <failure_type>
Confidence: <triage_confidence>
Severity:   <severity>

ROOT CAUSE
----------
<root_cause — expand into 2-3 sentences if needed, citing specific evidence from the excerpt below>

RECOMMENDED ACTION
------------------
<recommended_action — be specific, include commands or file references where relevant>

Raw excerpt for reference:
{state["raw_excerpt"]}
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        **state,
        "report": response.content,
    }