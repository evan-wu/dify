from typing import Any, Literal, Optional

from pydantic import BaseModel

from core.workflow.entities.base_node_data_entities import BaseNodeData
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


class CollectState(BaseModel):
    """
    Collect State.
    """
    collect_node_id: str
    current_runs: int

    class MetaData(BaseModel):
        """
        Data.
        """
        max_runs: int

    def get_current_runs(self) -> int:
        """
        Get current run times.
        """
        return self.current_runs

    metadata: MetaData
