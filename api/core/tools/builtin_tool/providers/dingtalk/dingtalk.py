from core.tools.builtin_tool.providers.dingtalk.tools.dingtalk_group_bot import DingTalkGroupBotTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class DingTalkProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        DingTalkGroupBotTool()
        pass
