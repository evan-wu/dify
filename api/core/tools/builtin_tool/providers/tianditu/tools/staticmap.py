import json
from collections.abc import Generator
from typing import Any, Union, Optional

import requests

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.builtin_tool.tool import BuiltinTool


class PoiSearchTool(BuiltinTool):
    def _invoke(
            self,
            user_id: str,
            tool_parameters: dict[str, Any],
            conversation_id: Optional[str] = None,
            app_id: Optional[str] = None,
            message_id: Optional[str] = None,
    ) -> ToolInvokeMessage | list[ToolInvokeMessage] | Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """

        geocoder_base_url = "http://api.tianditu.gov.cn/geocoder"
        base_url = "http://api.tianditu.gov.cn/staticimage"

        keyword = tool_parameters.get("keyword", "")
        if not keyword:
            return self.create_text_message("Invalid parameter keyword")

        tk = self.runtime.credentials["tianditu_api_key"]

        keyword_coords = requests.get(
            geocoder_base_url
            + "?ds="
            + json.dumps(
                {
                    "keyWord": keyword,
                },
                ensure_ascii=False,
            )
            + "&tk="
            + tk
        ).json()
        coords = keyword_coords["location"]["lon"] + "," + keyword_coords["location"]["lat"]

        result = requests.get(
            base_url + "?center=" + coords + "&markers=" + coords + "&width=400&height=300&zoom=14&tk=" + tk
        ).content

        return self.create_blob_message(
            blob=result, meta={"mime_type": "image/png"}, save_as=self.VariableKey.IMAGE.value
        )
