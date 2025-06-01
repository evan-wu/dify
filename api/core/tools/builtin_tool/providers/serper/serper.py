from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.serper.tools.serper_search import SerperSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class SerperProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            SerperSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={"query": "test", "result_type": "link"},
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
