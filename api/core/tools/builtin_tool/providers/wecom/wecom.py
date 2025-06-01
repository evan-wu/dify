from core.tools.builtin_tool.providers.wecom.tools.wecom_group_bot import WecomGroupBotTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class WecomProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        WecomGroupBotTool()
