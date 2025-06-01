from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.arxiv.tools.arxiv_search import ArxivSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class ArxivProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            ArxivSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "query": "John Doe",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
