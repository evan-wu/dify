from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.maths.tools.eval_expression import EvaluateExpressionTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class MathsProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            EvaluateExpressionTool().invoke(
                user_id="",
                tool_parameters={
                    "expression": "1+(2+3)*4",
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
