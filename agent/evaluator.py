import os
import json
import re
from langchain_openai import ChatOpenAI
from agent.state import AgentState

MAX_ITERATIONS = 3


def get_evaluator_llm():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def evaluator_node(state: AgentState) -> AgentState:
    """
    Critiques the triage classification using GPT-4o.
    Returns verdict: 'approve' or 'reject'.
    On reject, populates evaluator_feedback for the triage node to consider.
    On max iterations exceeded, sets failure_type to 'unclassified'.
    """
    iterations = state.get("eval_iterations", 0)

    # Hard stop — exceeded max iterations
    if iterations >= MAX_ITERATIONS:
        return {
            **state,
            "failure_type": "unclassified",
            "evaluator_feedback": None,
        }

    prompt = f"""
You are a senior software engineer auditing a CI test failure classification.

A triage system classified the following test failure excerpt. Your job is to
evaluate whether the classification is correct and well-reasoned.

--- TRIAGE OUTPUT ---
Failure type:  {state["failure_type"]}
Confidence:    {state["triage_confidence"]}
Reasoning:     {state["triage_reasoning"]}

--- TEST EXCERPT ---
{state["raw_excerpt"]}

--- CLASSIFICATION CRITERIA ---
- flaky: non-deterministic; requires evidence of timing variance, resource issues, or historical intermittency
- regression: requires explicit evidence of a prior passing state (commit reference, version, "previously")
- env_issue: missing dependency, import failure, binary not found, connection refused
- logic_bug: consistent deterministic failure; wrong output with no prior state or env evidence
- timeout: wall time exceeded, process killed, ReadTimeout

--- YOUR TASK ---
Evaluate the triage classification strictly. Ask yourself:
1. Does the excerpt actually contain the signals claimed in the reasoning?
2. Is there a more appropriate classification given the disambiguation rules?
3. Is the confidence level justified by the evidence in the excerpt?

Return JSON only:
{{
  "verdict": "<approve|reject>",
  "evaluation": "<two to three sentences explaining your verdict>",
  "suggested_type": "<the type you would assign if rejecting, or same type if approving>"
}}

Be strict. Approve only if the classification is clearly correct and well-supported
by specific evidence in the excerpt. Reject if there is meaningful ambiguity
or a better-supported alternative classification exists.
"""

    llm = get_evaluator_llm()
    response = llm.invoke(prompt)
    text = response.content

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        # If evaluator fails to parse, approve to avoid infinite loop
        return {**state, "eval_iterations": iterations + 1}

    result = json.loads(match.group())
    verdict = result.get("verdict", "approve")
    evaluation = result.get("evaluation", "")
    suggested_type = result.get("suggested_type", state["failure_type"])

    if verdict == "approve":
        return {
            **state,
            "eval_iterations": iterations + 1,
            "evaluator_feedback": None,
        }
    else:
        # Build actionable feedback for the triage node
        feedback = (
            f"Classification '{state['failure_type']}' was rejected.\n"
            f"Evaluator assessment: {evaluation}\n"
            f"Suggested type: {suggested_type}\n"
            f"Re-examine the excerpt carefully, specifically addressing these concerns."
        )
        return {
            **state,
            "eval_iterations": iterations + 1,
            "evaluator_feedback": feedback,
            "failure_type": state["failure_type"],  # preserve until triage revises
        }


def evaluator_route(state: AgentState) -> str:
    """
    Routing function called after the evaluator node.
    Returns the name of the next node.
    """
    # Max iterations exceeded — go straight to report
    if state.get("failure_type") == "unclassified":
        return "report"

    # Evaluator rejected — loop back to triage
    if state.get("evaluator_feedback"):
        return "triage"

    # Evaluator approved — route by failure type
    failure_type = state.get("failure_type")
    if failure_type == "flaky":
        return "flaky_analysis"
    return "analysis"