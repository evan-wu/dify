import logging
import time
from collections.abc import Generator, Mapping
from typing import Any

from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.enums import (
    ErrorStrategy,
    NodeExecutionType,
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
    execution_type = NodeExecutionType.RACE
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

        This method implements proper racing behavior with timeout support:
        1. Start monitoring for results immediately
        2. Wait for results with configurable timeout
        3. Complete when winning condition is met or timeout expires
        """
        logger.info(f"Starting race node {self._node_id} with strategy: {self._node_data.race_strategy}")

        # Start the race with timeout
        final_result = self._run_race_with_timeout()

        yield StreamCompletedEvent(node_run_result=final_result)

    def _run_race_with_timeout(self) -> NodeRunResult:
        """
        Run the race with proper timeout handling.

        This method implements the core racing logic:
        1. Monitor for results during the timeout period
        2. Complete early if winning condition is met
        3. Complete when timeout expires
        """
        start_time = time.time()
        timeout_seconds = self._node_data.timeout_seconds or 30.0
        collected_results = []

        logger.info("Starting race with timeout: %ss", timeout_seconds)

        while True:
            # Check for new results
            current_results = self._collect_available_results()

            # Add any new results we haven't seen before
            for result in current_results:
                if not any(r['selector'] == result['selector'] for r in collected_results):
                    collected_results.append(result)
                    logger.info(f"New result collected from {result['selector']}")

            # Check if we should complete early
            if self._should_complete_early(collected_results):
                winner = self._select_winner(collected_results)
                elapsed_time = time.time() - start_time
                logger.info(f"Race completed early after {elapsed_time:.2f}s")

                # Cancel losing branches
                self._cancel_losing_branches(winner, collected_results)

                return self._create_race_result(winner, collected_results, completed=True)

            # Check if timeout has expired
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_seconds:
                logger.info(f"Race timeout after {elapsed_time:.2f}s")
                return self._handle_timeout(collected_results)

            # Wait a bit before checking again (avoid busy waiting)
            time.sleep(0.1)

    def _should_complete_early(self, results: list[dict[str, Any]]) -> bool:
        """
        Determine if we should complete early based on the race strategy.
        """
        if not results:
            return False

        if self._node_data.race_strategy == RaceStrategy.FIRST_COMPLETE or self._node_data.race_strategy == RaceStrategy.FASTEST_VALID:
            return len(results) >= 1
        elif self._node_data.race_strategy == RaceStrategy.TIMEOUT_BEST:
            return False  # Always wait for timeout
        elif self._node_data.race_strategy == RaceStrategy.QUALITY_RACE:
            return len(results) >= self._node_data.max_winners
        return True

    def _handle_timeout(self, collected_results: list[dict[str, Any]]) -> NodeRunResult:
        """
        Handle timeout scenario - select best available result or fail.
        """
        if not collected_results:
            # No results collected during timeout
            if self._node_data.fail_on_timeout:
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.FAILED,
                    error="Race timeout with no results",
                    outputs={},
                    inputs={}
                )
            else:
                return NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs={'race_result': None, 'race_status': 'timeout'},
                    inputs={}
                )

        # Select best result from collected results
        winner = self._select_winner(collected_results)

        # Cancel losing branches on timeout as well
        self._cancel_losing_branches(winner, collected_results)

        return self._create_race_result(winner, collected_results, completed=False)

    def _cancel_losing_branches(self, winner: dict[str, Any], all_results: list[dict[str, Any]]) -> None:
        """
        Cancel the losing branches by storing cancellation information.

        Since we don't have direct access to the graph, we'll store the cancellation
        information in the node's state and let the workflow system handle it.
        """
        # Get all node IDs that should be cancelled
        nodes_to_cancel = []

        # Find all nodes that are not the winner
        for selector in self._node_data.variables:
            node_id = selector[0]  # First element is the node ID
            if node_id not in nodes_to_cancel:
                # Check if this node is not the winner
                is_winner = any(
                    result['selector'] == selector for result in all_results
                    if result == winner
                )
                if not is_winner:
                    nodes_to_cancel.append(node_id)

        if nodes_to_cancel:
            logger.info(f"Race node {self._node_id} determined winner, losing branches should be cancelled: {nodes_to_cancel}")

            # Store cancellation information in the node's outputs for now
            # This is a temporary solution until we implement proper cancellation
            self._cancelled_nodes = nodes_to_cancel
            logger.info("Marked nodes for cancellation: %s", nodes_to_cancel)
        else:
            self._cancelled_nodes = []

    def _collect_available_results(self) -> list[dict[str, Any]]:
        """
        Collect all currently available results from the variable pool.
        """
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
        return results

    def _select_winner(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Select the winner from available results based on strategy.
        """
        if not results:
            raise ValueError("No results available to select winner from")

        if self._node_data.race_strategy == RaceStrategy.FIRST_COMPLETE:
            return results[0]  # First available result
        elif self._node_data.race_strategy == RaceStrategy.FASTEST_VALID:
            return results[0]  # First valid result (already filtered)
        elif self._node_data.race_strategy == RaceStrategy.QUALITY_RACE:
            return self._select_best_quality_result(results)
        else:
            return results[0]  # Default to first

    def _create_race_result(self, winner: dict[str, Any], all_results: list[dict[str, Any]], completed: bool = True) -> NodeRunResult:
        """
        Create the final race result.
        """
        outputs = {
            'race_winner': winner['value'],
            'race_status': 'completed' if completed else 'timeout',
            'total_competitors': len(self._node_data.variables),
            'total_winners': len(all_results),
            'winner_selector': '.'.join(winner['selector'][1:]),  # Remove node_id prefix
            'cancelled_nodes': getattr(self, '_cancelled_nodes', [])  # Include cancellation info
        }

        inputs = {
            ".".join(winner['selector'][1:]): winner['value']
        }

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            outputs=outputs,
            inputs=inputs
        )

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

