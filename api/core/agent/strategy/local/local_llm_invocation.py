from dify_plugin.invocations.model.llm import LLMInvocation
from collections.abc import Generator
from typing import Literal, overload

from dify_plugin.core.entities.invocation import InvokeType
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool
)
from core.agent.strategy.local.local_to_plugin_types_convert import LocalToPluginTypesConvert
from core.model_runtime.model_providers import model_provider_factory
from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import (
    ModelType,
)


class LocalLLMInvocation(LLMInvocation):
    @overload
    def invoke(
            self,
            model_config: LLMModelConfig | dict,
            prompt_messages: list[PromptMessage],
            tools: list[PromptMessageTool] | None = None,
            stop: list[str] | None = None,
            stream: Literal[True] = True,
    ) -> Generator[LLMResultChunk, None, None]:
        ...

    @overload
    def invoke(
            self,
            model_config: LLMModelConfig | dict,
            prompt_messages: list[PromptMessage],
            tools: list[PromptMessageTool] | None = None,
            stop: list[str] | None = None,
            stream: Literal[False] = False,
    ) -> LLMResult:
        ...

    @overload
    def invoke(
            self,
            model_config: LLMModelConfig | dict,
            prompt_messages: list[PromptMessage],
            tools: list[PromptMessageTool] | None = None,
            stop: list[str] | None = None,
            stream: bool = True,
    ) -> Generator[LLMResultChunk, None, None] | LLMResult:
        ...

    def invoke(
            self,
            model_config: LLMModelConfig | dict,
            prompt_messages: list[PromptMessage],
            tools: list[PromptMessageTool] | None = None,
            stop: list[str] | None = None,
            stream: bool = True,
    ) -> Generator[LLMResultChunk, None, None] | LLMResult:
        """
        Invoke llm
        """
        if isinstance(model_config, dict):
            model_config = LLMModelConfig(**model_config)

        data = {
            **model_config.model_dump(),
            "prompt_messages": [message.model_dump() for message in prompt_messages],
            "tools": [tool.model_dump() for tool in tools] if tools else None,
            "stop": stop,
            "stream": stream,
        }

        if stream:
            model_manager = ModelManager()
            model_instance = model_manager.get_model_instance(
                tenant_id=model_provider_factory.tenant_id,
                model_type=ModelType.LLM,
                provider=model_config.provider,
                model=model_config.model
            )
            response = model_instance.invoke_llm(
                prompt_messages=LocalToPluginTypesConvert.to_local_prompt_message_type(list(prompt_messages)),
                model_parameters=model_config.completion_params,
                stop=list(stop or []),
                tools=tools,
                stream=True
            )

            return response

        result = LLMResult(
            model=model_config.model,
            message=AssistantPromptMessage(content=""),
            usage=LLMUsage.empty_usage(),
        )

        assert isinstance(result.message.content, str)

        for llm_result in self._backwards_invoke(
                InvokeType.LLM,
                LLMResultChunk,
                data,
        ):
            if isinstance(llm_result.delta.message.content, str):
                result.message.content += llm_result.delta.message.content
            if len(llm_result.delta.message.tool_calls) > 0:
                result.message.tool_calls = llm_result.delta.message.tool_calls
            if llm_result.delta.usage:
                result.usage.prompt_tokens += llm_result.delta.usage.prompt_tokens
                result.usage.completion_tokens += llm_result.delta.usage.completion_tokens
                result.usage.total_tokens += llm_result.delta.usage.total_tokens

                result.usage.completion_price = llm_result.delta.usage.completion_price
                result.usage.prompt_price = llm_result.delta.usage.prompt_price
                result.usage.total_price = llm_result.delta.usage.total_price
                result.usage.currency = llm_result.delta.usage.currency
                result.usage.latency = llm_result.delta.usage.latency

        return result
