import os
from langchain_anthropic import ChatAnthropic
from agent.state import AgentState


def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )


def report_node(state: AgentState) -> AgentState:
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