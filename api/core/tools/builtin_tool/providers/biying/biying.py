from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.biying.tools.biying_web_search import BiyingcnSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class BiyingProvider(BuiltinToolProviderController):
    def _validate_credentials(self, user_id: str, credentials: dict[str, Any]) -> None:
        try:
            BiyingcnSearchTool().fork_tool_runtime(
                meta={
                    "credentials": credentials,
                }
            ).invoke(
                user_id='',
                tool_parameters={
                    "query": "test",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))