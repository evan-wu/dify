from core.tools.builtin_tool.provider import BuiltinToolProviderController
from core.tools.utils.feishu_api_utils import auth


class FeishuWikiProvider(BuiltinToolProviderController):
    def _validate_credentials(self, user_id: str, credentials: dict) -> None:
        auth(credentials)
