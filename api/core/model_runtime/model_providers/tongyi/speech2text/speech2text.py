from typing import IO, Optional

import os
import tempfile
from http import HTTPStatus
import dashscope
from dashscope.audio.asr import *
from dashscope.common.error import *

from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.model_entities import AIModelEntity, FetchFrom, ModelType
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.speech2text_model import Speech2TextModel
from core.model_runtime.model_providers.tongyi._common import _CommonTongyi


class OpenAISpeech2TextModel(_CommonTongyi, Speech2TextModel):
    """
    Model class for OpenAI Speech to text model.
    """

    def _invoke(self, model: str, credentials: dict, file: IO[bytes], user: Optional[str] = None) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :param user: unique user id
        :return: text for given audio file
        """
        return self._speech2text_invoke(model, credentials, file)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            audio_file_path = self._get_demo_file_path()

            with open(audio_file_path, "rb") as audio_file:
                self._speech2text_invoke(model, credentials, audio_file)
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _speech2text_invoke(self, model: str, credentials: dict, file: IO[bytes]) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :return: text for given audio file
        """
        # transform credentials to kwargs for model instance
        credentials_kwargs = self._to_credential_kwargs(credentials)
        api_key = credentials_kwargs.get("dashscope_api_key")

        # init model client
        dashscope.api_key = api_key

        recognition = Recognition(
            model='paraformer-realtime-8k-v2',
            format='mp3',
            sample_rate=16000,
            language_hints=["zh", "en"],  # “language_hints”只支持paraformer-realtime-v2模型
            callback=None,
        )
        # save file to temp directory
        temp_file_path = tempfile.mktemp()
        with open(temp_file_path, "wb") as f:
            f.write(file.read())
        
        result = recognition.call(temp_file_path)
        os.remove(temp_file_path)

        if result.status_code != HTTPStatus.OK:
            raise RequestFailure(result.request_id, message=result.message)

        sentence_list = result.get_sentence()
        if sentence_list is None:
            return "No ASR result"
        else:
            print('The brief result is:  ')
            for sentence in sentence_list:
                print(sentence["text"])
            print(
                '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
                .format(
                    recognition.get_last_request_id(),
                    recognition.get_first_package_delay(),
                    recognition.get_last_package_delay(),
                ))
            return "".join(sentence["text"] for sentence in sentence_list)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> Optional[AIModelEntity]:
        """
        used to define customizable model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.SPEECH2TEXT,
            model_properties={},
            parameter_rules=[],
        )

        return entity
