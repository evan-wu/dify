from core.tools.builtin_tool.providers.slack.tools.slack_webhook import SlackWebhookTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class SlackProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        SlackWebhookTool()
        pass
