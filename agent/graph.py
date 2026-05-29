from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import triage_node

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.set_entry_point("triage")
    graph.add_edge("triage", END)
    return graph.compile()