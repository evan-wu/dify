from replicate.exceptions import ModelError, ReplicateError
from dify_plugin.errors.model import InvokeBadRequestError, InvokeError


class _CommonReplicate:
    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {InvokeBadRequestError: [ReplicateError, ModelError]}
