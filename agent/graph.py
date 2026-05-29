from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import triage_node
from agent.analysis import analysis_node
from agent.report import report_node


def route_by_failure_type(state: AgentState) -> str:
    return state["failure_type"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("triage")
    graph.add_conditional_edges(
        "triage",
        route_by_failure_type,
        {
            "flaky":      "analysis",
            "regression": "analysis",
            "env_issue":  "analysis",
            "logic_bug":  "analysis",
            "timeout":    "analysis",
        }
    )
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)

    return graph.compile()