from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.websearch.tools.web_search import WebSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class WebSearchAPIProvider(BuiltinToolProviderController):
    # validate when saving the api_key
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            WebSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={"query": "what is llm"},
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
