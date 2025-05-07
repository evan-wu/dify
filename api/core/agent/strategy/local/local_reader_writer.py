from typing import Generator

from dify_plugin.core.server.__base.request_reader import RequestReader
from dify_plugin.core.server.__base.response_writer import ResponseWriter


# empty implementation
class LocalRequestReader(RequestReader):
    def _read_stream(self) -> Generator["PluginInStream", None, None]:
        pass


class LocalResponseWriter(ResponseWriter):
    def write(self, data: str):
        pass

    def done(self):
        pass
