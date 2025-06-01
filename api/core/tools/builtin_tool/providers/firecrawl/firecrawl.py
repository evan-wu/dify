from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.builtin_tool.providers.firecrawl.tools.scrape import ScrapeTool
from core.tools.builtin_tool.provider import BuiltinToolProviderController


class FirecrawlProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict) -> None:
        try:
            # Example validation using the ScrapeTool, only scraping title for minimize content
            ScrapeTool().fork_tool_runtime(runtime={"credentials": credentials}).invoke(
                user_id="", tool_parameters={"url": "https://google.com", "onlyIncludeTags": "title"}
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
