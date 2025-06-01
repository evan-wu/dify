from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.devdocs.tools.searchDevDocs import SearchDevDocsTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class DevDocsProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            SearchDevDocsTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "doc": "python~3.12",
                    "topic": "library/code",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
