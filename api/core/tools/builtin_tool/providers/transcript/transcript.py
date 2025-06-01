from typing import Any

from core.tools.builtin_tool.provider import BuiltinToolProviderController


class YouTubeTranscriptProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        No credentials needed for YouTube Transcript API
        """
        pass
