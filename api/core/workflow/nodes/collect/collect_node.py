import json
import logging
from collections.abc import Generator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Optional, cast

from configs import dify_config
from core.model_runtime.utils.encoders import jsonable_encoder
from core.workflow.entities.node_entities import NodeRunMetadataKey, NodeRunResult
from core.workflow.entities.variable_pool import VariablePool
from core.workflow.graph_engine.entities.event import (
    BaseGraphEvent,
    BaseNodeEvent,
    BaseParallelBranchEvent,
    GraphRunFailedEvent,
    GraphRunSucceededEvent,
    InNodeEvent,
    IterationRunFailedEvent,
    IterationRunNextEvent,
    IterationRunStartedEvent,
    IterationRunSucceededEvent,
    NodeRunStreamChunkEvent,
    NodeRunSucceededEvent,
)
from core.workflow.graph_engine.entities.graph import Graph
from core.workflow.nodes.base import BaseNode
from core.workflow.nodes.collect.entities import CollectNodeData
from core.workflow.nodes.enums import NodeType
from core.workflow.nodes.event import NodeEvent, RunCompletedEvent
from core.workflow.nodes.if_else.entities import Condition
from core.workflow.utils.variable_template_parser import VariableTemplateParser
from extensions.ext_database import db
from models.workflow import Workflow, WorkflowNodeExecutionStatus, WorkflowRunningCollect

logger = logging.getLogger(__name__)


class CollectNode(BaseNode):
    """
    Collect Node.
    """
    _node_data_cls = CollectNodeData
    _node_type = NodeType.COLLECT
    VAR_NAME_CURRENT_RUNS = '_current_runs_'
    VAR_NAME_IS_RESUMED_COLLECT = '_is_resumed_collect_'

    def _run(self) -> Generator[NodeEvent | InNodeEvent, None, None]:
        """
        Run the node.
        """
        start_at = datetime.now(timezone.utc).replace(tzinfo=None)

        self.node_data = cast(CollectNodeData, self.node_data)
        variable_pool = self.graph_runtime_state.variable_pool
        max_runs = self.node_data.max_runs
        is_resumed_collect = variable_pool.get((self.node_id, CollectNode.VAR_NAME_IS_RESUMED_COLLECT))
        # first run: runs count from saved variables
        if not is_resumed_collect:
            current_runs = 1
            variable_pool.add((self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS), current_runs)
        else:
            current_runs = variable_pool.get((self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS)).value

        # reuse the iteration event for now
        yield IterationRunStartedEvent(
            iteration_id=self.id,
            iteration_node_id=self.node_id,
            iteration_node_type=self.node_type,
            iteration_node_data=self.node_data,
            start_at=start_at,
            inputs={},
            metadata={"iterator_length": 1},
            predecessor_node_id=self.previous_node_id,
        )

        yield IterationRunNextEvent(
            iteration_id=self.id,
            iteration_node_id=self.node_id,
            iteration_node_type=self.node_type,
            iteration_node_data=self.node_data,
            index=0,
            pre_iteration_output=None,
        )

        # start running inner graph
        from core.workflow.graph_engine.graph_engine import GraphEngine

        inner_start_node_id = self.node_data.start_node_id
        inner_graph = Graph.init(graph_config=self.graph_config, root_node_id=inner_start_node_id)
        if not inner_graph:
            raise ValueError("collect node inner graph not found")

        graph_engine = GraphEngine(
            tenant_id=self.tenant_id,
            app_id=self.app_id,
            workflow_type=self.workflow_type,
            workflow_id=self.workflow_id,
            user_id=self.user_id,
            user_from=self.user_from,
            invoke_from=self.invoke_from,
            call_depth=self.workflow_call_depth,
            graph=inner_graph,
            graph_config=self.graph_config,
            variable_pool=variable_pool,
            max_execution_steps=dify_config.WORKFLOW_MAX_EXECUTION_STEPS,
            max_execution_time=dify_config.WORKFLOW_MAX_EXECUTION_TIME,
            thread_pool_id=self.thread_pool_id,
        )

        rst = graph_engine.run()
        for event in rst:
            if isinstance(event, (BaseNodeEvent | BaseParallelBranchEvent)) and not event.in_iteration_id:
                event.in_iteration_id = self.node_id

            if (
                    isinstance(event, BaseNodeEvent)
                    and event.node_type == NodeType.ITERATION_START
                    and not isinstance(event, NodeRunStreamChunkEvent)
            ):
                continue

            if isinstance(event, NodeRunSucceededEvent):
                if event.route_node_state.node_run_result:
                    metadata = event.route_node_state.node_run_result.metadata
                    if not metadata:
                        metadata = {}

                    if NodeRunMetadataKey.ITERATION_ID not in metadata:
                        metadata[NodeRunMetadataKey.ITERATION_ID] = self.node_id
                        metadata[NodeRunMetadataKey.ITERATION_INDEX] = 0
                        event.route_node_state.node_run_result.metadata = metadata

                yield event
            elif isinstance(event, BaseGraphEvent):
                if isinstance(event, GraphRunFailedEvent):
                    # iteration run failed
                    yield IterationRunFailedEvent(
                        iteration_id=self.id,
                        iteration_node_id=self.node_id,
                        iteration_node_type=self.node_type,
                        iteration_node_data=self.node_data,
                        start_at=start_at,
                        inputs={},
                        outputs={"output": jsonable_encoder({})},
                        steps=1,
                        metadata={"total_tokens": graph_engine.graph_runtime_state.total_tokens},
                        error=event.error,
                    )

                    yield RunCompletedEvent(
                        run_result=NodeRunResult(
                            status=WorkflowNodeExecutionStatus.FAILED,
                            error=event.error,
                        )
                    )
                    return
            else:
                event = cast(InNodeEvent, event)
                yield event

        # check completed after run
        collect_node_output = None
        collect_completed = False
        if self.check_collect_completed(variable_pool, current_runs, max_runs):
            collect_completed = True
            # set collect node level output
            collect_node_output = variable_pool.get(
                self.node_data.output_selector
            )
            variable_pool.add(
                (self.node_id, 'output'),
                collect_node_output
            )

            # delete the saved collect state
            if is_resumed_collect:
                workflow = db.session.query(Workflow).filter(
                    Workflow.id == self.workflow_id
                ).first()
                self._delete_workflow_running_collect(workflow, variable_pool)

        # clear variables in current collect
        # remove all nodes outputs from variable pool
        for node_id in inner_graph.node_ids:
            variable_pool.remove((node_id,))  # the input is (node_id, [var_name])

        yield IterationRunSucceededEvent(
            iteration_id=self.id,
            iteration_node_id=self.node_id,
            iteration_node_type=self.node_type,
            iteration_node_data=self.node_data,
            start_at=start_at,
            inputs={},
            outputs={"output": jsonable_encoder(collect_node_output)},
            steps=1,
            metadata={"total_tokens": graph_engine.graph_runtime_state.total_tokens},
        )

        if collect_completed:
            yield RunCompletedEvent(
                run_result=NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs={"output": jsonable_encoder(collect_node_output)}
                )
            )
        else:
            current_runs += 1
            variable_pool.add((self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS), current_runs)

            # save reusable variables
            workflow = db.session.query(Workflow).filter(
                Workflow.id == self.workflow_id
            ).first()
            self._save_workflow_running_collect(workflow, variable_pool)

            # XXX: signal early exit the WHOLE workflow
            yield GraphRunSucceededEvent(outputs={})

    def _post_run_check_condition(self, variable_pool: VariablePool) -> bool:
        # post-check condition
        input_conditions, group_result = self.process_conditions(variable_pool, self.node_data.check_conditions)
        check_satisfied = all(group_result) if self.node_data.logical_operator == "and" else any(group_result)
        return check_satisfied

    def check_collect_completed(self, variable_pool: VariablePool, current_runs, max_runs):
        return current_runs >= max_runs or self._post_run_check_condition(variable_pool)

    def _save_workflow_running_collect(self, workflow: Workflow, variable_pool: VariablePool):
        # update or create
        conversation_id = variable_pool.get(('sys', 'conversation_id')).value
        collect_node_id = self.node_id
        current_runs = variable_pool.get(
                (self.node_id, CollectNode.VAR_NAME_CURRENT_RUNS)
        ).value
        variable_dict = {k: v for k, v in variable_pool.variable_dictionary.items()
                         if k != 'sys'}
        variable_dict_str = json.dumps(jsonable_encoder(variable_dict), ensure_ascii=False)

        running_collect = db.session.query(WorkflowRunningCollect).filter(
            WorkflowRunningCollect.tenant_id == workflow.tenant_id,
            WorkflowRunningCollect.app_id == workflow.app_id,
            WorkflowRunningCollect.workflow_id == workflow.id,
            WorkflowRunningCollect.workflow_version == workflow.version,
            WorkflowRunningCollect.conversation_id == conversation_id
        ).first()

        if not running_collect:
            running_collect = WorkflowRunningCollect(
                tenant_id=workflow.tenant_id,
                app_id=workflow.app_id,
                workflow_id=workflow.id,
                workflow_version=workflow.version,
                conversation_id=conversation_id,
                collect_node_id=collect_node_id,
                current_runs=current_runs,
                created_from=self.invoke_from.value,
                created_by=self.user_id,
                variable_dict=variable_dict_str,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            db.session.add(running_collect)
        else:
            running_collect.collect_node_id = collect_node_id
            running_collect.current_runs = current_runs
            running_collect.variable_dict = variable_dict_str
            running_collect.updated_at = workflow.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        db.session.commit()

    def _delete_workflow_running_collect(self, workflow: Workflow, variable_pool: VariablePool):
        conversation_id = variable_pool.get(('sys', 'conversation_id')).value

        db.session.query(WorkflowRunningCollect).filter(
            WorkflowRunningCollect.tenant_id == workflow.tenant_id,
            WorkflowRunningCollect.app_id == workflow.app_id,
            WorkflowRunningCollect.workflow_id == workflow.id,
            WorkflowRunningCollect.workflow_version == workflow.version,
            WorkflowRunningCollect.conversation_id == conversation_id
        ).delete()
        db.session.commit()

    @classmethod
    def _extract_variable_selector_to_variable_mapping(
            cls, graph_config: Mapping[str, Any], node_id: str, node_data: CollectNodeData
    ) -> Mapping[str, Sequence[str]]:
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
            actual_variable = variable_pool.get(condition.variable_selector)

            if condition.value is not None:
                variable_template_parser = VariableTemplateParser(template=condition.value)
                expected_value = variable_template_parser.extract_variable_selectors()
                variable_selectors = variable_template_parser.extract_variable_selectors()
                if variable_selectors:
                    for variable_selector in variable_selectors:
                        value = variable_pool.get(variable_selector.value_selector)
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
