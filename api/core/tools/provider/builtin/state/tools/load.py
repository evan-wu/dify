from typing import Any, Union

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool
from extensions.ext_database import db
from models.tools import StateToolConvState


class StateLoadTool(BuiltinTool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> Union[
        ToolInvokeMessage, list[ToolInvokeMessage]]:
        conv_id = tool_parameters.get('conversation_id')

        existing_state = db.session.query(StateToolConvState).filter(
            StateToolConvState.conversation_id == conv_id
        ).first()
        if existing_state:
            return self.create_text_message(existing_state.content)
        else:
            return self.create_text_message('{}')
