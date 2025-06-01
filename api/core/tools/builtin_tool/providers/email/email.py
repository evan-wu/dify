from core.tools.builtin_tool.providers.email.tools.send_mail import SendMailTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class SmtpProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        SendMailTool()
