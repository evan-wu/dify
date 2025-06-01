from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.provider import BuiltinToolProviderController
from core.tools.builtin_tool.providers.stackexchange.tools.searchStackExQuestions import SearchStackExQuestionsTool


class StackExchangeProvider(BuiltinToolProviderController):
    def _validate_credentials(self, user_id: str, credentials: dict) -> None:
        try:
            SearchStackExQuestionsTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).invoke(
                user_id="",
                tool_parameters={
                    "intitle": "Test",
                    "sort": "relevance",
                    "order": "desc",
                    "site": "stackoverflow",
                    "accepted": True,
                    "pagesize": 1,
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
