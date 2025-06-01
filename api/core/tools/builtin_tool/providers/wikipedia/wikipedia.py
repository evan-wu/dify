from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.wikipedia.tools.wikipedia_search import WikiPediaSearchTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class WikiPediaProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            WikiPediaSearchTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "query": "misaka mikoto",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
