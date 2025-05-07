from collections.abc import Generator

from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
    ToolPromptMessage,
    PromptMessage,
)
#
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage as LocalAssistantPromptMessage,
    PromptMessage as LocalPromptMessage,
    SystemPromptMessage as LocalSystemPromptMessage,
    ToolPromptMessage as LocalToolPromptMessage,
    UserPromptMessage as LocalUserPromptMessage,
)
from core.agent.entities import AgentInvokeMessage as LocalAgentInvokeMessage
from core.tools.entities.tool_entities import (
    ToolInvokeMessage as LocalToolInvokeMessage,
    ToolProviderType as LocalToolProviderType
)


class LocalToPluginTypesConvert:
    @classmethod
    def to_local_agent_invoke_message(cls, msgs: Generator[ToolInvokeMessage | AgentInvokeMessage]) \
            -> Generator[LocalAgentInvokeMessage]:
        for msg in msgs:
            local_msg = LocalAgentInvokeMessage(**msg.model_dump(mode="json"))
            yield local_msg

    @classmethod
    def to_local_prompt_message_type(cls, msgs: list[PromptMessage]) -> list[LocalPromptMessage]:
        local_msgs = []
        for msg in msgs:
            if isinstance(msg, SystemPromptMessage):
                local_msgs.append(LocalSystemPromptMessage(**msg.model_dump(mode="json")))
            elif isinstance(msg, AssistantPromptMessage):
                local_msgs.append(LocalAssistantPromptMessage(**msg.model_dump(mode="json")))
            elif isinstance(msg, UserPromptMessage):
                local_msgs.append(LocalUserPromptMessage(**msg.model_dump(mode="json")))
            elif isinstance(msg, ToolPromptMessage):
                local_msgs.append(LocalToolPromptMessage(**msg.model_dump(mode="json")))
            elif isinstance(msg, PromptMessage):
                local_msgs.append(LocalPromptMessage(**msg.model_dump(mode="json")))

        return local_msgs

    @classmethod
    def from_local_tool_invoke_message(cls, resp: Generator[LocalToolInvokeMessage]) -> Generator[ToolInvokeMessage]:
        for msg in resp:
            plugin_msg = ToolInvokeMessage(**msg.model_dump(mode="json"))
            yield plugin_msg

    @classmethod
    def to_local_tool_provider_type(cls, provider_type: ToolProviderType):
        local_provider_type = LocalToolProviderType.value_of(provider_type.value)
        return local_provider_type
