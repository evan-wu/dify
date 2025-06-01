from core.tools.builtin_tool.providers.feishu.tools.feishu_group_bot import FeishuGroupBotTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class FeishuProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        FeishuGroupBotTool()
