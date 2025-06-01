from core.tools.builtin_tool.provider import BuiltinToolProviderController
from core.tools.utils.feishu_api_utils import auth


class FeishuMessageProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        auth(credentials)
