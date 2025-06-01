from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.rapidapi.tools.google_news import GooglenewsTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class RapidapiProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            GooglenewsTool().fork_tool_runtime(
                meta={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "language_region": "en-US",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
