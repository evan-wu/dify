from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.youtube.tools.videos import YoutubeVideosAnalyticsTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class YahooFinanceProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            YoutubeVideosAnalyticsTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "channel": "UC2JZCsZSOudXA08cMMRCL9g",
                    "start_date": "2020-01-01",
                    "end_date": "2024-12-31",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
