from typing import Any

from core.tools.builtin_tool.provider import BuiltinToolProviderController
from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.tianditu.tools.poisearch import PoiSearchTool


class TiandituProvider(BuiltinToolProviderController):
    def _validate_credentials(self, user_id: str, credentials: dict[str, Any]) -> None:
        try:
            PoiSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "content": "北京",
                    "specify": "156110000",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
