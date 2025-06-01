from typing import Any

from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.stablediffusion.tools.stable_diffusion import StableDiffusionTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class StableDiffusionProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            StableDiffusionTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).validate_models()
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
