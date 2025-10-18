import json
import logging
from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core.model_runtime.utils.encoders import jsonable_encoder
from core.workflow.entities import VariablePool
from core.workflow.enums import (
    ErrorStrategy,
    NodeExecutionType,
    NodeType,
    WorkflowNodeExecutionMetadataKey,
    WorkflowNodeExecutionStatus,
)
from core.workflow.graph_events import (
    BaseGraphEvent,
    GraphNodeEventBase,
    GraphRunFailedEvent,
    GraphRunSucceededEvent,
)
from core.workflow.node_events import (
    IterationFailedEvent,
    IterationNextEvent,
    IterationStartedEvent,
    IterationSucceededEvent,
    NodeEventBase,
    NodeRunResult,
    StreamChunkEvent,
    StreamCompletedEvent,
)
from core.workflow.nodes.base.entities import BaseNodeData, RetryConfig
from core.workflow.nodes.base.node import Node
from core.workflow.nodes.collect.entities import CollectNodeData
from core.workflow.nodes.iteration.exc import (
    IterationGraphNotFoundError,
)
from core.workflow.utils.condition.processor import ConditionProcessor
from extensions.ext_database import db
from libs.datetime_utils import naive_utc_now
from models.workflow import Workflow, WorkflowRunningCollect

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CollectNode(Node):
    """
    Collect Node.
    """

    node_type = NodeType.COLLECT
    execution_type = NodeExecutionType.CONTAINER
    _node_data: CollectNodeData
    VAR_NAME_CURRENT_RUNS = '_current_runs_'
    VAR_NAME_IS_RESUMED_COLLECT = '_is_resumed_collect_'

    def init_node_data(self, data: Mapping[str, Any]):
        self._node_data = CollectNodeData.model_validate(data)

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
        return {
            "type": "collect"
        }

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self) -> Generator[GraphNodeEventBase | NodeEventBase, None, None]:  # type: ignore
        """
        Run the node.
        """
        started_at = naive_utc_now()

        variable_pool = self.graph_runtime_state.variable_pool
        max_runs = self._node_data.max_runs
        is_resumed_collect = variable_pool.get((self._node_id, CollectNode.VAR_NAME_IS_RESUMED_COLLECT))
        # first run: runs count from saved variables
        if not is_resumed_collect:
            current_runs = 1
            variable_pool.add((self._node_id, CollectNode.VAR_NAME_CURRENT_RUNS), current_runs)
        else:
            current_runs = variable_pool.get((self._node_id, CollectNode.VAR_NAME_CURRENT_RUNS)).value

        # reuse the iteration event for now
        yield IterationStartedEvent(
            start_at=started_at,
            inputs={},
            metadata={"iterator_length": 1}
        )

        yield IterationNextEvent(
            index=0
        )

        # start running inner graph
        graph_engine, inner_graph_node_ids = self._create_graph_engine()
        condition_processor = ConditionProcessor()

        rst = graph_engine.run()
        for event in rst:
            if isinstance(event, GraphNodeEventBase) and not event.in_iteration_id:
                event.in_iteration_id = self._node_id

            if (
                isinstance(event, GraphNodeEventBase)
                and event.node_type == NodeType.COLLECT_START
                and not isinstance(event, StreamChunkEvent)
            ):
                continue

            if isinstance(event, StreamCompletedEvent):
                if event.node_run_result:
                    metadata = event.metadata
                    if not metadata:
                        metadata = {}

                    if WorkflowNodeExecutionMetadataKey.ITERATION_ID not in metadata:
                        metadata[WorkflowNodeExecutionMetadataKey.ITERATION_ID] = self._node_id
                        metadata[WorkflowNodeExecutionMetadataKey.ITERATION_INDEX] = 0
                        event.route_node_state.node_run_result.metadata = metadata

                yield event
            elif isinstance(event, BaseGraphEvent):
                if isinstance(event, GraphRunFailedEvent):
                    # iteration run failed
                    yield IterationFailedEvent(
                        start_at=started_at,
                        inputs={},
                        outputs={"output": jsonable_encoder({})},
                        steps=1,
                        metadata={WorkflowNodeExecutionMetadataKey.TOTAL_TOKENS: graph_engine.graph_runtime_state.total_tokens},
                        error=event.error,
                    )

                    yield StreamCompletedEvent(
                        node_run_result=NodeRunResult(
                            status=WorkflowNodeExecutionStatus.FAILED,
                            error=event.error,
                        )
                    )
                    return
            else:
                yield event

        # check completed after run
        collect_node_output = None
        collect_completed = False
        if self.check_collect_completed(condition_processor, variable_pool, current_runs, max_runs):
            collect_completed = True
            # set collect node level output
            collect_node_output = variable_pool.get(
                self._node_data.output_selector
            )
            if collect_node_output:
                collect_node_output = collect_node_output.value
                variable_pool.add(
                    (self._node_id, 'output'),
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
        for node_id in inner_graph_node_ids:
            variable_pool.remove((node_id,))  # the input is (node_id, [var_name])

        yield IterationSucceededEvent(
            start_at=started_at,
            inputs={},
            outputs={"output": jsonable_encoder(collect_node_output)},
            steps=1,
            metadata={"total_tokens": graph_engine.graph_runtime_state.total_tokens},
        )

        if collect_completed:
            yield StreamCompletedEvent(
                node_run_result=NodeRunResult(
                    status=WorkflowNodeExecutionStatus.SUCCEEDED,
                    outputs={"output": jsonable_encoder(collect_node_output)}
                )
            )
        else:
            current_runs += 1
            variable_pool.add((self._node_id, CollectNode.VAR_NAME_CURRENT_RUNS), current_runs)

            # save reusable variables
            workflow = db.session.query(Workflow).filter(
                Workflow.id == self.workflow_id
            ).first()
            self._save_workflow_running_collect(workflow, variable_pool)

            # XXX: signal early exit the WHOLE workflow
            yield GraphRunSucceededEvent(outputs={})

    def _post_run_check_condition(self, condition_processor: ConditionProcessor, variable_pool: VariablePool) -> bool:
        # post-check condition
        _, _, check_satisfied = condition_processor.process_conditions(variable_pool=variable_pool,
                                                                       conditions=self._node_data.check_conditions,
                                                                       operator=self._node_data.logical_operator)
        return check_satisfied

    def check_collect_completed(self, condition_processor, variable_pool: VariablePool, current_runs, max_runs):
        return current_runs >= max_runs or self._post_run_check_condition(condition_processor, variable_pool)

    def _save_workflow_running_collect(self, workflow: Workflow, variable_pool: VariablePool):
        # update or create
        conversation_id = variable_pool.get(('sys', 'conversation_id')).value
        collect_node_id = self._node_id
        current_runs = variable_pool.get(
                (self._node_id, CollectNode.VAR_NAME_CURRENT_RUNS)
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
                created_at=datetime.now(UTC).replace(tzinfo=None)
            )
            db.session.add(running_collect)
        else:
            running_collect.collect_node_id = collect_node_id
            running_collect.current_runs = current_runs
            running_collect.variable_dict = variable_dict_str
            running_collect.updated_at = workflow.updated_at = datetime.now(UTC).replace(tzinfo=None)

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
        cls,
        *,
        graph_config: Mapping[str, Any],
        node_id: str,
        node_data: Mapping[str, Any],
    ) -> Mapping[str, Sequence[str]]:
        return {}

    def _append_iteration_info_to_event(
        self,
        event: GraphNodeEventBase,
        iter_run_index: int,
    ):
        event.in_iteration_id = self._node_id
        iter_metadata = {
            WorkflowNodeExecutionMetadataKey.ITERATION_ID: self._node_id,
            WorkflowNodeExecutionMetadataKey.ITERATION_INDEX: iter_run_index,
        }

        current_metadata = event.node_run_result.metadata
        if WorkflowNodeExecutionMetadataKey.ITERATION_ID not in current_metadata:
            event.node_run_result.metadata = {**current_metadata, **iter_metadata}

    def _create_graph_engine(self):
        # Import dependencies
        from core.workflow.entities import GraphInitParams, GraphRuntimeState
        from core.workflow.graph import Graph
        from core.workflow.graph_engine import GraphEngine
        from core.workflow.graph_engine.command_channels import InMemoryChannel
        from core.workflow.nodes.node_factory import DifyNodeFactory

        # Create GraphInitParams from node attributes
        graph_init_params = GraphInitParams(
            tenant_id=self.tenant_id,
            app_id=self.app_id,
            workflow_id=self.workflow_id,
            graph_config=self.graph_config,
            user_id=self.user_id,
            user_from=self.user_from.value,
            invoke_from=self.invoke_from.value,
            call_depth=self.workflow_call_depth,
        )
        # Create a deep copy of the variable pool for each iteration
        variable_pool_copy = self.graph_runtime_state.variable_pool.model_copy(deep=True)

        # Create a new GraphRuntimeState for this iteration
        graph_runtime_state_copy = GraphRuntimeState(
            variable_pool=variable_pool_copy,
            start_at=self.graph_runtime_state.start_at,
            total_tokens=0,
            node_run_steps=0,
        )

        # Create a new node factory with the new GraphRuntimeState
        node_factory = DifyNodeFactory(
            graph_init_params=graph_init_params, graph_runtime_state=graph_runtime_state_copy
        )

        # Initialize the iteration graph with the new node factory
        inner_graph = Graph.init(
            graph_config=self.graph_config, node_factory=node_factory, root_node_id=self._node_data.start_node_id
        )

        if not inner_graph:
            raise IterationGraphNotFoundError("collect node inner graph not found")

        # Create a new GraphEngine for this iteration
        graph_engine = GraphEngine(
            workflow_id=self.workflow_id,
            graph=inner_graph,
            graph_runtime_state=graph_runtime_state_copy,
            command_channel=InMemoryChannel(),  # Use InMemoryChannel for sub-graphs
        )

        return graph_engine, inner_graph.node_ids
