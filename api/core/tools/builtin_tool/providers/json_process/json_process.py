from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.json_process.tools.parse import JSONParseTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class JsonExtractProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            JSONParseTool().invoke(
                user_id="",
                tool_parameters={"content": '{"name": "John", "age": 30, "city": "New York"}', "json_filter": "$.name"},
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
