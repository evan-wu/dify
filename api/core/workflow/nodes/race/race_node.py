import logging
from collections.abc import Generator, Mapping
from typing import Any, Optional

from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.enums import (
    ErrorStrategy,
    NodeType,
)
from core.workflow.graph_events import (
    GraphNodeEventBase,
)
from core.workflow.node_events import (
    NodeEventBase,
    NodeRunResult,
    StreamCompletedEvent,
)
from core.workflow.nodes.base.entities import BaseNodeData, RetryConfig
from core.workflow.nodes.base.node import Node
from core.workflow.nodes.race.entities import RaceNodeData, RaceStrategy, WinCondition

logger = logging.getLogger(__name__)


class RaceNode(Node):
    """
    Race Node for competitive parallel execution.

    This node waits for parallel branches to complete and selects winners based
    on the configured race strategy (fastest, highest quality, etc.).
    """

    node_type = NodeType.RACE
    _node_data: RaceNodeData

    def init_node_data(self, data: Mapping[str, Any]):
        self._node_data = RaceNodeData.model_validate(data)

    def _get_error_strategy(self) -> ErrorStrategy | None:
        return self._node_data.error_strategy

    def _get_retry_config(self) -> RetryConfig:
        return self._node_data.retry_config

    def _get_title(self) -> str:
        return self._node_data.title

    def _get_description(self) -> str | None:
        return self._node_data.desc

    def _get_default_value_dict(self) -> dict[str, Any]:
        return self._node_data.default_value_dict

    def get_base_node_data(self) -> BaseNodeData:
        return self._node_data

    @classmethod
    def get_default_config(cls, filters: Mapping[str, object] | None = None) -> Mapping[str, object]:
        """
        Get default configuration for the race node.
        """
        return {
            "type": "race",
            "config": {
                "race_strategy": "first_complete",
                "win_condition": "any_result",
                "timeout_seconds": 30.0,
                "max_winners": 1,
                "variables": [],
                "fail_on_timeout": False,
                "fail_on_all_errors": True
            }
        }

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self) -> Generator[GraphNodeEventBase | NodeEventBase, None, None]:  # type: ignore
        """
        Run the race node with competitive parallel execution.
        """
        logger.info(f"Starting race node {self._node_id} with strategy: {self._node_data.race_strategy}")

        # Collect all available variables from the completed branches
        results = []
        for selector in self._node_data.variables:
            variable = self.graph_runtime_state.variable_pool.get(selector)
            if variable is not None:
                result = {
                    'selector': selector,
                    'variable': variable,
                    'value': variable.to_object()
                }
                if self._is_winning_result(result):
                    results.append(result)

        # Process results based on strategy
        final_result = self._process_race_results(results)

        yield StreamCompletedEvent(node_run_result=final_result)

    def _is_winning_result(self, result: dict[str, Any]) -> bool:
        """
        Determine if a result meets the winning condition.
        """
        if self._node_data.win_condition == WinCondition.ANY_RESULT:
            return result['value'] is not None

        elif self._node_data.win_condition == WinCondition.NO_ERROR:
            # Check if the result indicates an error (this is simplified)
            value = result['value']
            if isinstance(value, dict) and value.get('error'):
                return False
            return value is not None

        elif self._node_data.win_condition == WinCondition.CUSTOM_VALIDATION:
            return self._custom_validate_result(result)

        return True

    def _custom_validate_result(self, result: dict[str, Any]) -> bool:
        """
        Apply custom validation logic to determine if result is valid.
        """
        if not self._node_data.validation_expression:
            return True

        try:
            # This is a simplified implementation
            # In a real system, you'd want a safe expression evaluator
            value = result['value']
            # For demo purposes, just check if value has certain properties
            if isinstance(value, dict):
                return bool(value.get('success', True))
            return bool(value)
        except Exception as e:
            logger.warning("Custom validation failed: %s", e)
            return False

    def _process_race_results(self, winners: list[dict[str, Any]]) -> NodeRunResult:
        """
        Process the race results and return the final node result.
        """
        if not winners:
            # No winners found
            if self._node_data.fail_on_timeout:
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.FAILED,
                    error="Race timeout with no winners",
                    outputs={},
                    inputs={}
                )
            else:
                # Try to get any available result as fallback
                fallback_result = self._get_fallback_result()
                if fallback_result:
                    return fallback_result

                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs={'race_result': None, 'race_status': 'timeout'},
                    inputs={}
                )

        # Process winners based on strategy
        if self._node_data.race_strategy == RaceStrategy.FIRST_COMPLETE:
            winner = winners[0]
        elif self._node_data.race_strategy == RaceStrategy.QUALITY_RACE:
            winner = self._select_best_quality_result(winners)
        else:
            winner = winners[0]  # Default to first

        # Prepare outputs
        outputs = {
            'race_winner': winner['value'],
            'race_status': 'completed',
            'total_competitors': len(self._node_data.variables),
            'total_winners': len(winners)
        }

        inputs = {
            ".".join(winner['selector'][1:]): winner['value']
        }

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            outputs=outputs,
            inputs=inputs
        )

    def _select_best_quality_result(self, winners: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Select the best quality result from winners.
        """
        if not self._node_data.scoring_expression:
            # Default to fastest (first winner)
            return winners[0]

        best_winner = winners[0]
        best_score = 0

        for winner in winners:
            try:
                score = self._score_result(winner)
                if score > best_score:
                    best_score = score
                    best_winner = winner
            except Exception as e:
                logger.warning("Scoring failed for result: %s", e)

        return best_winner

    def _score_result(self, result: dict[str, Any]) -> float:
        """
        Score a result based on the scoring expression.
        """
        # Simplified scoring implementation
        # In a real system, you'd want a safe expression evaluator
        value = result['value']

        if isinstance(value, dict):
            # Example: score based on response length, quality indicators, etc.
            score = 0.0
            if 'confidence' in value:
                score += float(value['confidence'])
            if 'text' in value:
                score += len(str(value['text'])) * 0.001  # Slight preference for longer responses
            return score

        return 1.0  # Default score

    def _get_fallback_result(self) -> Optional[NodeRunResult]:
        """
        Get any available result as a fallback when no winners are found.
        """
        for selector in self._node_data.variables:
            variable = self.graph_runtime_state.variable_pool.get(selector)
            if variable is not None:
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs={
                        'race_result': variable.to_object(),
                        'race_status': 'fallback'
                    },
                    inputs={".".join(selector[1:]): variable.to_object()}
                )

        return None
