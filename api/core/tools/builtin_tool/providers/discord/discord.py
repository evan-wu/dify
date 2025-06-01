from typing import Any

from core.tools.builtin_tool.providers.discord.tools.discord_webhook import DiscordWebhookTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class DiscordProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        DiscordWebhookTool()
