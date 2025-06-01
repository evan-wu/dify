from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.crossref.tools.query_doi import CrossRefQueryDOITool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class CrossRefProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            CrossRefQueryDOITool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "doi": "10.1007/s00894-022-05373-8",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
