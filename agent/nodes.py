from langchain_anthropic import ChatAnthropic
from agent.state import AgentState

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def triage_node(state: AgentState) -> AgentState:
    prompt = f"""
You are a test failure triage system. Classify the following test failure excerpt.

Failure types:
- flaky: passes sometimes, fails sometimes; look for: intermittent, ResourceWarning, random seed, timing variance, historical pass rate
- regression: was passing before, now fails; look for: previously, used to, before commit, expected X got Y where X was known-good
- env_issue: missing dependency, wrong environment; look for: ImportError, ModuleNotFoundError, ConnectionRefused, binary not found
- logic_bug: consistent deterministic failure; look for: AssertionError with fixed values, wrong output, no prior passing state
- timeout: exceeded time limit; look for: wall time, process killed, ReadTimeout, exceeded limit

Disambiguation rules:
- flaky requires evidence of non-determinism — if the excerpt shows only a single consistent failure with no timing/resource language, prefer logic_bug
- regression requires explicit evidence of a prior passing state — without it, prefer logic_bug
- env_issue takes priority over logic_bug when an ImportError or connection error is present

Return JSON only:
{{
  "failure_type": "<one of the five types>",
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence citing the specific signal that determined your choice>"
}}

Test excerpt:
{state["raw_excerpt"]}
"""
    response = llm.invoke(prompt)
    
    import json, re
    text = response.content
    # Extract the last JSON block — after the reasoning steps
    matches = re.findall(r'\{[^{}]*"failure_type"[^{}]*\}', text)
    if not matches:
        raise ValueError(f"No JSON found in response:\n{text}")

    result = json.loads(matches[-1])  # take the last one — after the reasoning
    
    return {
        **state,
        "failure_type": result["failure_type"],
        "triage_confidence": result["confidence"],
    }