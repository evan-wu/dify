from collections.abc import Sequence
from typing import Optional, cast

from core.workflow.entities.base_node_data_entities import BaseNodeData
from core.workflow.entities.node_entities import NodeType
from core.workflow.entities.variable_pool import VariablePool
from core.workflow.nodes.base_node import BaseNode
from core.workflow.nodes.collect.entities import CollectNodeData, CollectState
from core.workflow.nodes.if_else.entities import Condition
from core.workflow.utils.variable_template_parser import VariableTemplateParser


class CollectNode(BaseNode):
    """
    Collect Node.
    """
    _node_data_cls = CollectNodeData
    _node_type = NodeType.COLLECT
    VAR_NAME_CURRENT_RUNS = '_current_runs_'
    VAR_NAME_IS_RESUMED_COLLECT = '_is_resumed_collect_'

    def _run(self, variable_pool: VariablePool) -> CollectState:
        """
        Run the node.
        """
        self.node_data = cast(CollectNodeData, self.node_data)
        max_runs = self.node_data.max_runs

        # first run: runs count from saved variables
        if not variable_pool.get((self.node_id, CollectNode.VAR_NAME_IS_RESUMED_COLLECT)):
            current_runs = 0
        else:
            current_runs = variable_pool.get((self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS)).value

        state = CollectState(collect_node_id=self.node_id,
                             current_runs=current_runs,
                             metadata=CollectState.MetaData(max_runs=max_runs))
        
        self._set_current_run_variable(variable_pool, state)
        return state

    def _post_run_check_condition(self, variable_pool: VariablePool) -> bool:
        # post-check condition
        input_conditions, group_result = self.process_conditions(variable_pool, self.node_data.check_conditions)
        check_satisfied = all(group_result) if self.node_data.logical_operator == "and" else any(group_result)
        return check_satisfied

    def _set_current_run_variable(self, variable_pool: VariablePool, state: CollectState):
        variable_pool.add((self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS), state.current_runs)

    def get_next_run(self, variable_pool: VariablePool, state: CollectState) -> Optional[str]:
        """
        Get next collect run start node id based on the graph.
        """
        # move to next iteration, if is resumed collect, let it run once
        is_resumed_collect = variable_pool.get((self.node_id, CollectNode.VAR_NAME_IS_RESUMED_COLLECT))
        if not is_resumed_collect:
            state.current_runs += 1
            self._set_current_run_variable(variable_pool, state)

        node_data = cast(CollectNodeData, self.node_data)
        if self.check_collect_completed(variable_pool, state):
            return None

        # for resumed collect, ++
        if is_resumed_collect:
            state.current_runs += 1
            self._set_current_run_variable(variable_pool, state)

        return node_data.start_node_id

    def check_collect_completed(self, variable_pool: VariablePool, state: CollectState):
        return self._reached_runs_limit(state) or self._post_run_check_condition(variable_pool)

    def _reached_runs_limit(self, state: CollectState):
        """
        Check if iteration limit is reached.
        :return: True if iteration limit is reached, False otherwise
        """
        return state.current_runs >= state.metadata.max_runs

    @classmethod
    def _extract_variable_selector_to_variable_mapping(cls, node_data: BaseNodeData) -> dict[str, list[str]]:
        """
        Extract variable selector to variable mapping
        :param node_data: node data
        :return:
        """
        return {}

    """ Copy from if-else-node """

    def process_conditions(self, variable_pool: VariablePool, conditions: Sequence[Condition]):
        input_conditions = []
        group_result = []

        for condition in conditions:
            actual_variable = variable_pool.get_any(condition.variable_selector)

            if condition.value is not None:
                variable_template_parser = VariableTemplateParser(template=condition.value)
                expected_value = variable_template_parser.extract_variable_selectors()
                variable_selectors = variable_template_parser.extract_variable_selectors()
                if variable_selectors:
                    for variable_selector in variable_selectors:
                        value = variable_pool.get_any(variable_selector.value_selector)
                        expected_value = variable_template_parser.format({variable_selector.variable: value})
                else:
                    expected_value = condition.value
            else:
                expected_value = None

            comparison_operator = condition.comparison_operator
            input_conditions.append(
                {
                    "actual_value": actual_variable,
                    "expected_value": expected_value,
                    "comparison_operator": comparison_operator
                }
            )

            result = self.evaluate_condition(actual_variable, expected_value, comparison_operator)
            group_result.append(result)

        return input_conditions, group_result

    def evaluate_condition(
            self, actual_value: Optional[str | list], expected_value: str, comparison_operator: str
    ) -> bool:
        """
        Evaluate condition
        :param actual_value: actual value
        :param expected_value: expected value
        :param comparison_operator: comparison operator

        :return: bool
        """
        if comparison_operator == "contains":
            return self._assert_contains(actual_value, expected_value)
        elif comparison_operator == "not contains":
            return self._assert_not_contains(actual_value, expected_value)
        elif comparison_operator == "start with":
            return self._assert_start_with(actual_value, expected_value)
        elif comparison_operator == "end with":
            return self._assert_end_with(actual_value, expected_value)
        elif comparison_operator == "is":
            return self._assert_is(actual_value, expected_value)
        elif comparison_operator == "is not":
            return self._assert_is_not(actual_value, expected_value)
        elif comparison_operator == "empty":
            return self._assert_empty(actual_value)
        elif comparison_operator == "not empty":
            return self._assert_not_empty(actual_value)
        elif comparison_operator == "=":
            return self._assert_equal(actual_value, expected_value)
        elif comparison_operator == "≠":
            return self._assert_not_equal(actual_value, expected_value)
        elif comparison_operator == ">":
            return self._assert_greater_than(actual_value, expected_value)
        elif comparison_operator == "<":
            return self._assert_less_than(actual_value, expected_value)
        elif comparison_operator == "≥":
            return self._assert_greater_than_or_equal(actual_value, expected_value)
        elif comparison_operator == "≤":
            return self._assert_less_than_or_equal(actual_value, expected_value)
        elif comparison_operator == "null":
            return self._assert_null(actual_value)
        elif comparison_operator == "not null":
            return self._assert_not_null(actual_value)
        else:
            raise ValueError(f"Invalid comparison operator: {comparison_operator}")

    def _assert_contains(self, actual_value: Optional[str | list], expected_value: str) -> bool:
        """
        Assert contains
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if not actual_value:
            return False

        if not isinstance(actual_value, str | list):
            raise ValueError('Invalid actual value type: string or array')

        if expected_value not in actual_value:
            return False
        return True

    def _assert_not_contains(self, actual_value: Optional[str | list], expected_value: str) -> bool:
        """
        Assert not contains
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if not actual_value:
            return True

        if not isinstance(actual_value, str | list):
            raise ValueError('Invalid actual value type: string or array')

        if expected_value in actual_value:
            return False
        return True

    def _assert_start_with(self, actual_value: Optional[str], expected_value: str) -> bool:
        """
        Assert start with
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if not actual_value:
            return False

        if not isinstance(actual_value, str):
            raise ValueError('Invalid actual value type: string')

        if not actual_value.startswith(expected_value):
            return False
        return True

    def _assert_end_with(self, actual_value: Optional[str], expected_value: str) -> bool:
        """
        Assert end with
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if not actual_value:
            return False

        if not isinstance(actual_value, str):
            raise ValueError('Invalid actual value type: string')

        if not actual_value.endswith(expected_value):
            return False
        return True

    def _assert_is(self, actual_value: Optional[str], expected_value: str) -> bool:
        """
        Assert is
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, str):
            raise ValueError('Invalid actual value type: string')

        if actual_value != expected_value:
            return False
        return True

    def _assert_is_not(self, actual_value: Optional[str], expected_value: str) -> bool:
        """
        Assert is not
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, str):
            raise ValueError('Invalid actual value type: string')

        if actual_value == expected_value:
            return False
        return True

    def _assert_empty(self, actual_value: Optional[str]) -> bool:
        """
        Assert empty
        :param actual_value: actual value
        :return:
        """
        if not actual_value:
            return True
        return False

    def _assert_not_empty(self, actual_value: Optional[str]) -> bool:
        """
        Assert not empty
        :param actual_value: actual value
        :return:
        """
        if actual_value:
            return True
        return False

    def _assert_equal(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert equal
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value != expected_value:
            return False
        return True

    def _assert_not_equal(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert not equal
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value == expected_value:
            return False
        return True

    def _assert_greater_than(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert greater than
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value <= expected_value:
            return False
        return True

    def _assert_less_than(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert less than
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value >= expected_value:
            return False
        return True

    def _assert_greater_than_or_equal(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert greater than or equal
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value < expected_value:
            return False
        return True

    def _assert_less_than_or_equal(self, actual_value: Optional[int | float], expected_value: str) -> bool:
        """
        Assert less than or equal
        :param actual_value: actual value
        :param expected_value: expected value
        :return:
        """
        if actual_value is None:
            return False

        if not isinstance(actual_value, int | float):
            raise ValueError('Invalid actual value type: number')

        if isinstance(actual_value, int):
            expected_value = int(expected_value)
        else:
            expected_value = float(expected_value)

        if actual_value > expected_value:
            return False
        return True

    def _assert_null(self, actual_value: Optional[int | float]) -> bool:
        """
        Assert null
        :param actual_value: actual value
        :return:
        """
        if actual_value is None:
            return True
        return False

    def _assert_not_null(self, actual_value: Optional[int | float]) -> bool:
        """
        Assert not null
        :param actual_value: actual value
        :return:
        """
        if actual_value is not None:
            return True
        return False