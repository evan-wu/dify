from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.bing.tools.bing_web_search import BingSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class BingProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            BingSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).validate_credentials(
                credentials=credentials,
                tool_parameters={
                    "query": "test",
                    "result_type": "link",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
