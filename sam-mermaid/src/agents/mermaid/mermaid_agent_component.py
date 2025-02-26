"""The agent component for the mermaid"""

import os
import copy
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

from .actions.mermaid_draw import DrawAction

info = copy.deepcopy(agent_info)
info["agent_name"] = "mermaid"
info["class_name"] = "MermaidAgentComponent"
info["description"] = (
    "This agent handles requests to generate and render diagrams or visualizations using the Mermaid.js syntax."
)

class MermaidAgentComponent(BaseAgentComponent):
    info = info
    actions = [DrawAction]
