from datetime import datetime, timezone
from typing import Any, Union

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool
from extensions.ext_database import db
from models.tools import StateToolConvState


class StateSaveTool(BuiltinTool):
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any]) -> Union[
        ToolInvokeMessage, list[ToolInvokeMessage]]:
        conv_id = tool_parameters.get('conversation_id')
        content = tool_parameters.get('content')

        existing_state = db.session.query(StateToolConvState).filter(
            StateToolConvState.conversation_id == conv_id
        ).first()
        if existing_state:
            existing_state.content = content
            existing_state.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            new_state = StateToolConvState(
                conversation_id=conv_id,
                content=content,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            db.session.add(new_state)
        db.session.commit()
        return self.create_text_message('success')
