from core.tools.builtin_tool.provider import BuiltinToolProviderController
from core.tools.utils.lark_api_utils import lark_auth


class LarkBaseProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        lark_auth(credentials)
