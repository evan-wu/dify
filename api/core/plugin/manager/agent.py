from collections.abc import Generator, Callable
from typing import Any, Optional, TypeVar

from core.agent.entities import AgentInvokeMessage
from core.agent.strategy.local.agent_strategy_factory import AgentStrategyFactory
from core.plugin.entities.plugin import GenericProviderID
from core.plugin.entities.plugin_daemon import (
    PluginAgentProviderEntity,
)
from dify_plugin.core.entities.plugin.request import AgentInvokeRequest

import json
from pydantic import BaseModel
from core.plugin.entities.plugin_daemon import PluginDaemonBasicResponse, PluginDaemonError, PluginDaemonInnerError
T = TypeVar("T", bound=(BaseModel | dict | list | bool | str))


class PluginAgentManager:
    def fetch_agent_strategy_providers(self, tenant_id: str) -> list[PluginAgentProviderEntity]:
        """
        Fetch agent providers for the given tenant.
        """

        def transformer(json_response: dict[str, Any]) -> dict:
            for provider in json_response.get("data", []):
                declaration = provider.get("declaration", {}) or {}
                provider_name = declaration.get("identity", {}).get("name")
                for strategy in declaration.get("strategies", []):
                    strategy["identity"]["provider"] = provider_name

            return json_response

        response = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/agent_strategies",
            list[PluginAgentProviderEntity],
            params={"page": 1, "page_size": 256},
            transformer=transformer,
        )

        for provider in response:
            provider.declaration.identity.name = f"{provider.plugin_id}/{provider.declaration.identity.name}"

            # override the provider name for each tool to plugin_id/provider_name
            for strategy in provider.declaration.strategies:
                strategy.identity.provider = provider.declaration.identity.name

        return response

    def fetch_agent_strategy_provider(self, tenant_id: str, provider: str) -> PluginAgentProviderEntity:
        """
        Fetch tool provider for the given tenant and plugin.
        """
        agent_provider_id = GenericProviderID(provider)

        def transformer(json_response: dict[str, Any]) -> dict:
            # skip if error occurs
            if json_response.get("data") is None or json_response.get("data", {}).get("declaration") is None:
                return json_response

            for strategy in json_response.get("data", {}).get("declaration", {}).get("strategies", []):
                strategy["identity"]["provider"] = agent_provider_id.provider_name

            return json_response

        response = self._request_with_plugin_daemon_response(
            "GET",
            f"plugin/{tenant_id}/management/agent_strategy",
            PluginAgentProviderEntity,
            params={"provider": agent_provider_id.provider_name, "plugin_id": agent_provider_id.plugin_id},
            transformer=transformer,
        )

        response.declaration.identity.name = f"{response.plugin_id}/{response.declaration.identity.name}"

        # override the provider name for each tool to plugin_id/provider_name
        for strategy in response.declaration.strategies:
            strategy.identity.provider = response.declaration.identity.name

        return response

    def invoke(
        self,
        tenant_id: str,
        user_id: str,
        agent_provider: str,
        agent_strategy: str,
        agent_params: dict[str, Any],
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> Generator[AgentInvokeMessage, None, None]:
        """
        Invoke the agent with the given tenant, user, plugin, provider, name and parameters.
        """
        agent_strategy_factory = AgentStrategyFactory()
        response = agent_strategy_factory.invoke_agent_strategy(AgentInvokeRequest(**{
                "user_id": user_id,
                "conversation_id": conversation_id,
                "app_id": app_id,
                "message_id": message_id,
                "agent_strategy_provider": agent_provider,
                "agent_strategy": agent_strategy,
                "agent_strategy_params": agent_params,
            }))
        return response

    def _request_with_plugin_daemon_response(
        self,
        method: str,
        path: str,
        type: type[T],
        headers: dict | None = None,
        data: bytes | dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        transformer: Callable[[dict], dict] | None = None,
    ) -> T:
        """
        Make a request to the plugin daemon inner API and return the response as a model.
        """
        # TODO: use local agent-strategies data from agent_strategy_factory
        if "agent_strategies" in path:
            json_response = json.loads('''
        {
    "code": 0,
    "message": "success",
    "data": [
        {
            "id": "609f2469-9bb5-4201-9d3f-9d2dd0d40c6d",
            "created_at": "2025-04-26T05:11:53.4202Z",
            "updated_at": "2025-04-26T05:11:53.4202Z",
            "tenant_id": "697e5a4b-81b6-4e07-9928-e6443446925b",
            "provider": "agent",
            "plugin_unique_identifier": "langgenius/agent:0.0.14@26958a0e80a10655ce73812bdb7c35a66ce7b16f5ac346d298bda17ff85efd1e",
            "plugin_id": "langgenius/agent",
            "declaration": {
                "identity": {
                    "author": "langgenius",
                    "name": "agent",
                    "description": {
                        "en_US": "Agent",
                        "zh_Hans": "Agent",
                        "pt_BR": "Agent"
                    },
                    "icon": "e74e644589f5d78cd6019be7b92050c2b54b2645139af705fe610649a73282cf.svg",
                    "label": {
                        "en_US": "Agent",
                        "zh_Hans": "Agent",
                        "pt_BR": "Agent"
                    },
                    "tags": []
                },
                "strategies": [
                    {
                        "identity": {
                            "author": "Dify",
                            "name": "function_calling",
                            "label": {
                                "en_US": "FunctionCalling",
                                "zh_Hans": "FunctionCalling",
                                "pt_BR": "FunctionCalling"
                            }
                        },
                        "description": {
                            "en_US": "Function Calling is a basic strategy for agent, model will use the tools provided to perform the task.",
                            "zh_Hans": "Function Calling 是一个基本的 Agent 策略，模型将使用提供的工具来执行任务。",
                            "pt_BR": "Function Calling is a basic strategy for agent, model will use the tools provided to perform the task."
                        },
                        "parameters": [
                            {
                                "name": "model",
                                "label": {
                                    "en_US": "Model",
                                    "zh_Hans": "模型",
                                    "pt_BR": "Model"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "model-selector",
                                "auto_generate": null,
                                "template": null,
                                "scope": "tool-call&llm",
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "tools",
                                "label": {
                                    "en_US": "Tool list",
                                    "zh_Hans": "工具列表",
                                    "pt_BR": "Tool list"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "array[tools]",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "instruction",
                                "label": {
                                    "en_US": "Instruction",
                                    "zh_Hans": "指令",
                                    "pt_BR": "Instruction"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "string",
                                "auto_generate": {
                                    "type": "prompt_instruction"
                                },
                                "template": {
                                    "enabled": true
                                },
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "query",
                                "label": {
                                    "en_US": "Query",
                                    "zh_Hans": "查询",
                                    "pt_BR": "Query"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "string",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "maximum_iterations",
                                "label": {
                                    "en_US": "Maximum Iterations",
                                    "zh_Hans": "最大迭代次数",
                                    "pt_BR": "Maximum Iterations"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "number",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": 3,
                                "min": 1,
                                "max": 30,
                                "precision": null,
                                "options": null
                            }
                        ],
                        "output_schema": null,
                        "features": [
                            "history-messages"
                        ]
                    },
                    {
                        "identity": {
                            "author": "Dify",
                            "name": "ReAct",
                            "label": {
                                "en_US": "ReAct",
                                "zh_Hans": "ReAct",
                                "pt_BR": "ReAct"
                            }
                        },
                        "description": {
                            "en_US": "ReAct is a basic strategy for agent, model will use the tools provided to perform the task.",
                            "zh_Hans": "ReAct 是一个基本的 Agent 策略，模型将使用提供的工具来执行任务。",
                            "pt_BR": "ReAct is a basic strategy for agent, model will use the tools provided to perform the task."
                        },
                        "parameters": [
                            {
                                "name": "model",
                                "label": {
                                    "en_US": "Model",
                                    "zh_Hans": "模型",
                                    "pt_BR": "Model"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "model-selector",
                                "auto_generate": null,
                                "template": null,
                                "scope": "tool-call&llm",
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "tools",
                                "label": {
                                    "en_US": "Tool list",
                                    "zh_Hans": "工具列表",
                                    "pt_BR": "Tool list"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "array[tools]",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "instruction",
                                "label": {
                                    "en_US": "Instruction",
                                    "zh_Hans": "指令",
                                    "pt_BR": "Instruction"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "string",
                                "auto_generate": {
                                    "type": "prompt_instruction"
                                },
                                "template": {
                                    "enabled": true
                                },
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "query",
                                "label": {
                                    "en_US": "Query",
                                    "zh_Hans": "查询",
                                    "pt_BR": "Query"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "string",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": null,
                                "min": null,
                                "max": null,
                                "precision": null,
                                "options": null
                            },
                            {
                                "name": "maximum_iterations",
                                "label": {
                                    "en_US": "Maximum Iterations",
                                    "zh_Hans": "最大迭代次数",
                                    "pt_BR": "Maximum Iterations"
                                },
                                "help": {
                                    "en_US": ""
                                },
                                "type": "number",
                                "auto_generate": null,
                                "template": null,
                                "scope": null,
                                "required": true,
                                "default": 3,
                                "min": 1,
                                "max": 30,
                                "precision": null,
                                "options": null
                            }
                        ],
                        "output_schema": null,
                        "features": [
                            "history-messages"
                        ]
                    }
                ]
            }
        }
    ]
}
        ''')
        elif "agent_strategy" in path:
            json_response = json.loads('''
            {
    "code": 0,
    "message": "success",
    "data": {
        "id": "609f2469-9bb5-4201-9d3f-9d2dd0d40c6d",
        "created_at": "2025-04-26T05:11:53.4202Z",
        "updated_at": "2025-04-26T05:11:53.4202Z",
        "tenant_id": "697e5a4b-81b6-4e07-9928-e6443446925b",
        "provider": "agent",
        "plugin_unique_identifier": "langgenius/agent:0.0.14@26958a0e80a10655ce73812bdb7c35a66ce7b16f5ac346d298bda17ff85efd1e",
        "plugin_id": "langgenius/agent",
        "declaration": {
            "identity": {
                "author": "langgenius",
                "name": "agent",
                "description": {
                    "en_US": "Agent",
                    "zh_Hans": "Agent",
                    "pt_BR": "Agent"
                },
                "icon": "e74e644589f5d78cd6019be7b92050c2b54b2645139af705fe610649a73282cf.svg",
                "label": {
                    "en_US": "Agent",
                    "zh_Hans": "Agent",
                    "pt_BR": "Agent"
                },
                "tags": []
            },
            "strategies": [
                {
                    "identity": {
                        "author": "Dify",
                        "name": "function_calling",
                        "label": {
                            "en_US": "FunctionCalling",
                            "zh_Hans": "FunctionCalling",
                            "pt_BR": "FunctionCalling"
                        }
                    },
                    "description": {
                        "en_US": "Function Calling is a basic strategy for agent, model will use the tools provided to perform the task.",
                        "zh_Hans": "Function Calling 是一个基本的 Agent 策略，模型将使用提供的工具来执行任务。",
                        "pt_BR": "Function Calling is a basic strategy for agent, model will use the tools provided to perform the task."
                    },
                    "parameters": [
                        {
                            "name": "model",
                            "label": {
                                "en_US": "Model",
                                "zh_Hans": "模型",
                                "pt_BR": "Model"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "model-selector",
                            "auto_generate": null,
                            "template": null,
                            "scope": "tool-call&llm",
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "tools",
                            "label": {
                                "en_US": "Tool list",
                                "zh_Hans": "工具列表",
                                "pt_BR": "Tool list"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "array[tools]",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "instruction",
                            "label": {
                                "en_US": "Instruction",
                                "zh_Hans": "指令",
                                "pt_BR": "Instruction"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "string",
                            "auto_generate": {
                                "type": "prompt_instruction"
                            },
                            "template": {
                                "enabled": true
                            },
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "query",
                            "label": {
                                "en_US": "Query",
                                "zh_Hans": "查询",
                                "pt_BR": "Query"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "string",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "maximum_iterations",
                            "label": {
                                "en_US": "Maximum Iterations",
                                "zh_Hans": "最大迭代次数",
                                "pt_BR": "Maximum Iterations"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "number",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": 3,
                            "min": 1,
                            "max": 30,
                            "precision": null,
                            "options": null
                        }
                    ],
                    "output_schema": null,
                    "features": [
                        "history-messages"
                    ]
                },
                {
                    "identity": {
                        "author": "Dify",
                        "name": "ReAct",
                        "label": {
                            "en_US": "ReAct",
                            "zh_Hans": "ReAct",
                            "pt_BR": "ReAct"
                        }
                    },
                    "description": {
                        "en_US": "ReAct is a basic strategy for agent, model will use the tools provided to perform the task.",
                        "zh_Hans": "ReAct 是一个基本的 Agent 策略，模型将使用提供的工具来执行任务。",
                        "pt_BR": "ReAct is a basic strategy for agent, model will use the tools provided to perform the task."
                    },
                    "parameters": [
                        {
                            "name": "model",
                            "label": {
                                "en_US": "Model",
                                "zh_Hans": "模型",
                                "pt_BR": "Model"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "model-selector",
                            "auto_generate": null,
                            "template": null,
                            "scope": "tool-call&llm",
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "tools",
                            "label": {
                                "en_US": "Tool list",
                                "zh_Hans": "工具列表",
                                "pt_BR": "Tool list"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "array[tools]",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "instruction",
                            "label": {
                                "en_US": "Instruction",
                                "zh_Hans": "指令",
                                "pt_BR": "Instruction"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "string",
                            "auto_generate": {
                                "type": "prompt_instruction"
                            },
                            "template": {
                                "enabled": true
                            },
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "query",
                            "label": {
                                "en_US": "Query",
                                "zh_Hans": "查询",
                                "pt_BR": "Query"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "string",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": null,
                            "min": null,
                            "max": null,
                            "precision": null,
                            "options": null
                        },
                        {
                            "name": "maximum_iterations",
                            "label": {
                                "en_US": "Maximum Iterations",
                                "zh_Hans": "最大迭代次数",
                                "pt_BR": "Maximum Iterations"
                            },
                            "help": {
                                "en_US": ""
                            },
                            "type": "number",
                            "auto_generate": null,
                            "template": null,
                            "scope": null,
                            "required": true,
                            "default": 3,
                            "min": 1,
                            "max": 30,
                            "precision": null,
                            "options": null
                        }
                    ],
                    "output_schema": null,
                    "features": [
                        "history-messages"
                    ]
                }
            ]
        }
    }
}
            ''')

        if transformer:
            json_response = transformer(json_response)

        rep = PluginDaemonBasicResponse[type](**json_response)  # type: ignore

        return rep.data
