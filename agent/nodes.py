import os
import json
import re
from langchain_anthropic import ChatAnthropic
from agent.state import AgentState

def get_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_retries=3, 
    )

def triage_node(state: AgentState) -> AgentState:
    feedback = state.get("evaluator_feedback")
    iterations = state.get("eval_iterations", 0)

    # Build the feedback block only when revising
    feedback_block = ""
    if feedback:
        feedback_block = f"""
EVALUATOR FEEDBACK (iteration {iterations}):
{feedback}

Reconsider your classification in light of the above criticism.
Explain specifically how you are addressing each point raised.
"""

    prompt = f"""
You are a test failure triage system. Classify the following test failure excerpt.

Failure types:
- flaky: passes sometimes, fails sometimes; look for: intermittent, ResourceWarning, random seed, timing variance, historical pass rate
- regression: was passing before, now fails; look for: previously, used to, before commit, expected X got Y where X was known-good
- env_issue: missing dependency, wrong environment; look for: ImportError, ModuleNotFoundError, ConnectionRefused, binary not found
- logic_bug: consistent deterministic failure; look for: AssertionError with fixed values, wrong output, no prior passing state
- timeout: exceeded time limit; look for: wall time, process killed, ReadTimeout, exceeded limit

Disambiguation rules:
- flaky requires evidence of non-determinism — a single consistent failure with no timing/resource language is logic_bug
- regression requires explicit evidence of a prior passing state — without it, prefer logic_bug
- env_issue takes priority over logic_bug when an ImportError or connection error is present
{feedback_block}
Return JSON only:
{{
  "failure_type": "<one of the five types>",
  "confidence": "<high|medium|low>",
  "reasoning": "<two to three sentences explaining the specific signals that determined your classification>"
}}

Test excerpt:
{state["raw_excerpt"]}
"""

    llm = get_llm()
    response = llm.invoke(prompt)
    text = response.content

    matches = re.findall(r'\{[^{}]*"failure_type"[^{}]*\}', text, re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON found in triage response:\n{text}")

    result = json.loads(matches[-1])

    return {
        **state,
        "failure_type": result["failure_type"],
        "triage_confidence": result["confidence"],
        "triage_reasoning": result.get("reasoning", ""),
        "evaluator_feedback": None,  # clear after each triage pass
    }