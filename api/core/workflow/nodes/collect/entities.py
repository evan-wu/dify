from typing import Any, Literal, Optional

from core.workflow.nodes.base.entities import BaseNodeData
from core.workflow.nodes.if_else.entities import Condition


class CollectNodeData(BaseNodeData):
    """
    Collect Node Data.
    """
    start_node_id: str
    max_runs: int
    logical_operator: Optional[Literal["and", "or"]] = "and"
    check_conditions: Optional[list[Condition]] = None

    output_selector: list[str]  # output selector
    output: Optional[Any] = None
    OUTPUT_NAME: str = 'output'
