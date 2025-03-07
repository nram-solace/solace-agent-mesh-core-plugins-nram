"""The agent component for the rag"""

import os
import copy
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from solace_agent_mesh.agents.base_agent_component import (
    agent_info,
    BaseAgentComponent,
)

# Sample action import
from rag.actions.sample_action import SampleAction

info = copy.deepcopy(agent_info)
info["agent_name"] = "rag"
info["class_name"] = "RagAgentComponent"
info["description"] = (
    "The agent scans documents of resources, ingest documents in a vector database and "
    "provides relevant information when a user explicitly requests it."
)


class RagAgentComponent(BaseAgentComponent):
    info = info
    # sample action
    actions = [SampleAction]
