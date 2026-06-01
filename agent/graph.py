from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import triage_node
from agent.analysis import analysis_node, flaky_analysis_node
from agent.report import report_node
from agent.evaluator import evaluator_node, evaluator_route


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("flaky_analysis", flaky_analysis_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("triage")

    # Triage always goes to evaluator first
    graph.add_edge("triage", "evaluator")

    # Evaluator routes conditionally
    graph.add_conditional_edges(
        "evaluator",
        evaluator_route,
        {
            "triage":         "triage",
            "flaky_analysis": "flaky_analysis",
            "analysis":       "analysis",
            "report":         "report",
        }
    )

    graph.add_edge("flaky_analysis", "report")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)

    return graph.compile()