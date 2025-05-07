from collections.abc import Generator
from typing import Any

from dify_plugin.core.runtime import BackwardsInvocation
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType

from core.app.entities.app_invoke_entities import InvokeFrom
from core.agent.strategy.local.local_to_plugin_types_convert import LocalToPluginTypesConvert


class LocalToolInvocation(BackwardsInvocation[ToolInvokeMessage]):
    def invoke_builtin_tool(
        self, provider: str, tool_name: str, parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke builtin tool
        """
        return self.invoke(ToolProviderType.BUILT_IN, provider, tool_name, parameters)

    def invoke_workflow_tool(
        self, provider: str, tool_name: str, parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke workflow tool
        """
        return self.invoke(ToolProviderType.WORKFLOW, provider, tool_name, parameters)

    def invoke_api_tool(
        self, provider: str, tool_name: str, parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke api tool
        """
        return self.invoke(ToolProviderType.API, provider, tool_name, parameters)

    def invoke(
        self,
        provider_type: ToolProviderType,
        provider: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke tool
        """
        from core.tools.tool_manager import ToolManager
        from core.model_runtime.model_providers import model_provider_factory

        local_provider_type = LocalToPluginTypesConvert.to_local_tool_provider_type(provider_type)
        tool = ToolManager.get_tool_runtime(local_provider_type, provider, tool_name, model_provider_factory.tenant_id,
                                            invoke_from=InvokeFrom.WEB_APP)
        resp = tool.invoke(user_id="local_agent_tool_invocation",
                           tool_parameters=parameters)

        return LocalToPluginTypesConvert.from_local_tool_invoke_message(resp)
