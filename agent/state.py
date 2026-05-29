from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    raw_excerpt: str
    test_name: str
    runner: str
    timestamp: str

    # Triage node output
    failure_type: Optional[str]      # flaky | regression | env_issue | logic_bug | timeout
    triage_confidence: Optional[str] # high | medium | low

    # Analysis node output
    root_cause_hypothesis: Optional[str]
    recommended_action: Optional[str]

    # Report node output
    report: Optional[str]