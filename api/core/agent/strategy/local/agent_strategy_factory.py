import os
from abc import ABC
from concurrent.futures import ThreadPoolExecutor

from dify_plugin import AgentStrategy
from dify_plugin.core.entities.plugin.request import AgentInvokeRequest
from dify_plugin.core.runtime import Session
from dify_plugin.core.utils.class_loader import load_single_subclass_from_source
from dify_plugin.entities.agent import AgentStrategyProviderConfiguration
from dify_plugin.core.utils.yaml_loader import load_yaml_file

from core.agent.strategy.local.local_llm_invocation import LocalLLMInvocation
from core.agent.strategy.local.local_reader_writer import LocalRequestReader, LocalResponseWriter
from core.agent.strategy.local.local_tool_invocation import LocalToolInvocation


class AgentStrategyFactory(ABC):
    def __init__(self):
        self.agent_strategies_configuration = []
        self.agent_strategies_mapping = {}

        current_path = os.path.abspath(__file__)
        cot_agent_path = os.path.dirname(current_path)

        # TODO: list dir and add agent config
        strategy_config_file_path = os.path.join(cot_agent_path, "..", "cot_agent", "provider", "agent.yaml")
        fs = load_yaml_file(strategy_config_file_path)
        for i in range(len(fs.get("strategies", []))):
            fs.get("strategies")[i] = os.path.join(cot_agent_path, "..", "cot_agent", fs.get("strategies")[i])
        cot_agent_strategies_configuration = AgentStrategyProviderConfiguration(**fs)
        self.agent_strategies_configuration.append(cot_agent_strategies_configuration)
        self._resolve_agent_providers()

    def _resolve_agent_providers(self):
        """
        walk through all the agent providers and strategies and load the classes from sources
        """
        for provider in self.agent_strategies_configuration:
            strategies = {}
            for strategy in provider.strategies:
                strategy_source = strategy.extra.python.source
                strategy_module_source = os.path.splitext(strategy_source)[0]
                strategy_module_source = strategy_module_source.replace("/", ".")
                strategy_cls = load_single_subclass_from_source(
                    module_name=f"core.agent.strategy.cot_agent.{strategy_module_source}",  # TODO: only for cot_agent
                    script_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cot_agent", strategy_source),
                    parent_type=AgentStrategy,
                )

                strategies[strategy.identity.name] = (strategy, strategy_cls)

            self.agent_strategies_mapping[provider.identity.name] = (provider, strategies)

    def get_agent_strategy_cls(self, provider: str, agent: str):
        """
        get the agent class by provider
        :param provider: provider name
        :param agent: agent name
        :return: agent class
        """
        for provider_registration in self.agent_strategies_mapping:
            if provider_registration == provider:
                registration = self.agent_strategies_mapping[provider_registration][1].get(agent)
                if registration:
                    return registration[1]

    def invoke_agent_strategy(self, request: AgentInvokeRequest):
        provider = request.agent_strategy_provider
        if "/" in provider:
            provider = provider.rsplit("/", 1)[1]
        agent_cls = self.get_agent_strategy_cls(provider, request.agent_strategy)
        if agent_cls is None:
            raise ValueError(
                f"Agent `{request.agent_strategy}` not found for provider `{request.agent_strategy_provider}`"
            )
        local_session = Session(session_id="",
                                executor=ThreadPoolExecutor(),
                                reader=LocalRequestReader(),
                                writer=LocalResponseWriter(),
                                install_method=None,
                                dify_plugin_daemon_url=None,
                                )
        local_session.model.llm = LocalLLMInvocation()
        local_session.tool = LocalToolInvocation()
        agent = agent_cls(session=local_session)
        yield from agent.invoke(request.agent_strategy_params)


if __name__ == '__main__':
    factory = AgentStrategyFactory()
    req = AgentInvokeRequest(
        user_id='test123',
        agent_strategy_provider="agent",
        agent_strategy="function_calling",
        agent_strategy_params={
            'model': {'provider': 'tongyi', 'model': 'qwen-plus', 'model_type': 'llm', 'mode': 'chat', 'completion_params': {}, 'type': 'model-selector', 'history_prompt_messages': [], 'entity': {'model': 'qwen-plus', 'label': {'zh_Hans': 'qwen-plus', 'en_US': 'qwen-plus'}, 'model_type': 'llm', 'features': ['multi-tool-call', 'agent-thought', 'stream-tool-call'], 'fetch_from': 'predefined-model', 'model_properties': {'mode': 'chat', 'context_size': 128000}, 'deprecated': False, 'parameter_rules': [{'name': 'temperature', 'use_template': 'temperature', 'label': {'zh_Hans': '温度', 'en_US': 'Temperature'}, 'type': 'float', 'help': {'zh_Hans': '用于控制随机性和多样性的程度。具体来说，temperature值控制了生成文本时对每个候选词的概率分布进行平滑的程度。较高的temperature值会降低概率分布的峰值，使得更多的低概率词被选择，生成结果更加多样化；而较低的temperature值则会增强概率分布的峰值，使得高概率词更容易被选择，生成结果更加确定。', 'en_US': 'Used to control the degree of randomness and diversity. Specifically, the temperature value controls the degree to which the probability distribution of each candidate word is smoothed when generating text. A higher temperature value will reduce the peak value of the probability distribution, allowing more low-probability words to be selected, and the generated results will be more diverse; while a lower temperature value will enhance the peak value of the probability distribution, making it easier for high-probability words to be selected. , the generated results are more certain.'}, 'required': False, 'default': 0.3, 'min': 0.0, 'max': 2.0, 'precision': 2, 'options': []}, {'name': 'max_tokens', 'use_template': 'max_tokens', 'label': {'zh_Hans': '最大标记', 'en_US': 'Max Tokens'}, 'type': 'int', 'help': {'zh_Hans': '用于指定模型在生成内容时token的最大数量，它定义了生成的上限，但不保证每次都会生成到这个数量。', 'en_US': 'It is used to specify the maximum number of tokens when the model generates content. It defines the upper limit of generation, but does not guarantee that this number will be generated every time.'}, 'required': False, 'default': 8192, 'min': 1.0, 'max': 8192.0, 'precision': 0, 'options': []}, {'name': 'top_p', 'use_template': 'top_p', 'label': {'zh_Hans': 'Top P', 'en_US': 'Top P'}, 'type': 'float', 'help': {'zh_Hans': '生成过程中核采样方法概率阈值，例如，取值为0.8时，仅保留概率加起来大于等于0.8的最可能token的最小集合作为候选集。取值范围为（0,1.0)，取值越大，生成的随机性越高；取值越低，生成的确定性越高。', 'en_US': 'The probability threshold of the kernel sampling method during the generation process. For example, when the value is 0.8, only the smallest set of the most likely tokens with a sum of probabilities greater than or equal to 0.8 is retained as the candidate set. The value range is (0,1.0). The larger the value, the higher the randomness generated; the lower the value, the higher the certainty generated.'}, 'required': False, 'default': 0.8, 'min': 0.1, 'max': 0.9, 'precision': 2, 'options': []}, {'name': 'top_k', 'use_template': None, 'label': {'zh_Hans': '取样数量', 'en_US': 'Top k'}, 'type': 'int', 'help': {'zh_Hans': '生成时，采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。', 'en_US': 'The size of the sample candidate set when generated. For example, when the value is 50, only the 50 highest-scoring tokens in a single generation form a randomly sampled candidate set. The larger the value, the higher the randomness generated; the smaller the value, the higher the certainty generated.'}, 'required': False, 'default': None, 'min': 0.0, 'max': 99.0, 'precision': None, 'options': []}, {'name': 'seed', 'use_template': None, 'label': {'zh_Hans': '随机种子', 'en_US': 'Random seed'}, 'type': 'int', 'help': {'zh_Hans': '生成时使用的随机数种子，用户控制模型生成内容的随机性。支持无符号64位整数，默认值为 1234。在使用seed时，模型将尽可能生成相同或相似的结果，但目前不保证每次生成的结果完全相同。', 'en_US': 'The random number seed used when generating, the user controls the randomness of the content generated by the model. Supports unsigned 64-bit integers, default value is 1234. When using seed, the model will try its best to generate the same or similar results, but there is currently no guarantee that the results will be exactly the same every time.'}, 'required': False, 'default': 1234, 'min': None, 'max': None, 'precision': None, 'options': []}, {'name': 'repetition_penalty', 'use_template': None, 'label': {'zh_Hans': '重复惩罚', 'en_US': 'Repetition penalty'}, 'type': 'float', 'help': {'zh_Hans': '用于控制模型生成时的重复度。提高repetition_penalty时可以降低模型生成的重复度。1.0表示不做惩罚。', 'en_US': 'Used to control the repeatability when generating models. Increasing repetition_penalty can reduce the duplication of model generation. 1.0 means no punishment.'}, 'required': False, 'default': 1.1, 'min': None, 'max': None, 'precision': None, 'options': []}, {'name': 'enable_search', 'use_template': None, 'label': {'zh_Hans': '联网搜索', 'en_US': 'Web Search'}, 'type': 'boolean', 'help': {'zh_Hans': '模型内置了互联网搜索服务，该参数控制模型在生成文本时是否参考使用互联网搜索结果。启用互联网搜索，模型会将搜索结果作为文本生成过程中的参考信息，但模型会基于其内部逻辑“自行判断”是否使用互联网搜索结果。', 'en_US': 'The model has a built-in Internet search service. This parameter controls whether the model refers to Internet search results when generating text. When Internet search is enabled, the model will use the search results as reference information in the text generation process, but the model will "judge" whether to use Internet search results based on its internal logic.'}, 'required': False, 'default': False, 'min': None, 'max': None, 'precision': None, 'options': []}, {'name': 'response_format', 'use_template': 'response_format', 'label': {'zh_Hans': '回复格式', 'en_US': 'Response Format'}, 'type': 'string', 'help': {'zh_Hans': '设置一个返回格式，确保 llm 的输出尽可能是有效的代码块，如 JSON、XML 等', 'en_US': 'Set a response format, ensure the output from llm is a valid code block as possible, such as JSON, XML, etc.'}, 'required': False, 'default': None, 'min': None, 'max': None, 'precision': None, 'options': ['JSON', 'XML']}], 'pricing': {'input': '0.0008', 'output': '0.002', 'unit': '0.001', 'currency': 'RMB'}}}, 'tools': [{'identity': {'author': 'Dify', 'name': 'current_time', 'label': {'en_US': 'Current Time', 'zh_Hans': '获取当前时间', 'pt_BR': 'Current Time', 'ja_JP': 'Current Time'}, 'provider': 'time', 'icon': None}, 'parameters': [{'name': 'format', 'label': {'en_US': 'Format', 'zh_Hans': '格式', 'pt_BR': 'Format', 'ja_JP': 'Format'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': False, 'default': '%Y-%m-%d %H:%M:%S', 'min': None, 'max': None, 'precision': None, 'options': [], 'type': 'string', 'human_description': {'en_US': 'Time format in strftime standard.', 'zh_Hans': 'strftime 标准的时间格式。', 'pt_BR': 'Time format in strftime standard.', 'ja_JP': 'Time format in strftime standard.'}, 'form': 'form', 'llm_description': None}, {'name': 'timezone', 'label': {'en_US': 'Timezone', 'zh_Hans': '时区', 'pt_BR': 'Timezone', 'ja_JP': 'Timezone'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': False, 'default': 'UTC', 'min': None, 'max': None, 'precision': None, 'options': [{'value': 'UTC', 'label': {'en_US': 'UTC', 'zh_Hans': 'UTC', 'pt_BR': 'UTC', 'ja_JP': 'UTC'}}, {'value': 'America/New_York', 'label': {'en_US': 'America/New_York', 'zh_Hans': '美洲/纽约', 'pt_BR': 'America/New_York', 'ja_JP': 'America/New_York'}}, {'value': 'America/Los_Angeles', 'label': {'en_US': 'America/Los_Angeles', 'zh_Hans': '美洲/洛杉矶', 'pt_BR': 'America/Los_Angeles', 'ja_JP': 'America/Los_Angeles'}}, {'value': 'America/Chicago', 'label': {'en_US': 'America/Chicago', 'zh_Hans': '美洲/芝加哥', 'pt_BR': 'America/Chicago', 'ja_JP': 'America/Chicago'}}, {'value': 'America/Sao_Paulo', 'label': {'en_US': 'America/Sao_Paulo', 'zh_Hans': '美洲/圣保罗', 'pt_BR': 'América/São Paulo', 'ja_JP': 'America/Sao_Paulo'}}, {'value': 'Asia/Shanghai', 'label': {'en_US': 'Asia/Shanghai', 'zh_Hans': '亚洲/上海', 'pt_BR': 'Asia/Shanghai', 'ja_JP': 'Asia/Shanghai'}}, {'value': 'Asia/Ho_Chi_Minh', 'label': {'en_US': 'Asia/Ho_Chi_Minh', 'zh_Hans': '亚洲/胡志明市', 'pt_BR': 'Ásia/Ho Chi Minh', 'ja_JP': 'Asia/Ho_Chi_Minh'}}, {'value': 'Asia/Tokyo', 'label': {'en_US': 'Asia/Tokyo', 'zh_Hans': '亚洲/东京', 'pt_BR': 'Asia/Tokyo', 'ja_JP': 'Asia/Tokyo'}}, {'value': 'Asia/Dubai', 'label': {'en_US': 'Asia/Dubai', 'zh_Hans': '亚洲/迪拜', 'pt_BR': 'Asia/Dubai', 'ja_JP': 'Asia/Dubai'}}, {'value': 'Asia/Kolkata', 'label': {'en_US': 'Asia/Kolkata', 'zh_Hans': '亚洲/加尔各答', 'pt_BR': 'Asia/Kolkata', 'ja_JP': 'Asia/Kolkata'}}, {'value': 'Asia/Seoul', 'label': {'en_US': 'Asia/Seoul', 'zh_Hans': '亚洲/首尔', 'pt_BR': 'Asia/Seoul', 'ja_JP': 'Asia/Seoul'}}, {'value': 'Asia/Singapore', 'label': {'en_US': 'Asia/Singapore', 'zh_Hans': '亚洲/新加坡', 'pt_BR': 'Asia/Singapore', 'ja_JP': 'Asia/Singapore'}}, {'value': 'Europe/London', 'label': {'en_US': 'Europe/London', 'zh_Hans': '欧洲/伦敦', 'pt_BR': 'Europe/London', 'ja_JP': 'Europe/London'}}, {'value': 'Europe/Berlin', 'label': {'en_US': 'Europe/Berlin', 'zh_Hans': '欧洲/柏林', 'pt_BR': 'Europe/Berlin', 'ja_JP': 'Europe/Berlin'}}, {'value': 'Europe/Moscow', 'label': {'en_US': 'Europe/Moscow', 'zh_Hans': '欧洲/莫斯科', 'pt_BR': 'Europe/Moscow', 'ja_JP': 'Europe/Moscow'}}, {'value': 'Australia/Sydney', 'label': {'en_US': 'Australia/Sydney', 'zh_Hans': '澳大利亚/悉尼', 'pt_BR': 'Australia/Sydney', 'ja_JP': 'Australia/Sydney'}}, {'value': 'Pacific/Auckland', 'label': {'en_US': 'Pacific/Auckland', 'zh_Hans': '太平洋/奥克兰', 'pt_BR': 'Pacific/Auckland', 'ja_JP': 'Pacific/Auckland'}}, {'value': 'Africa/Cairo', 'label': {'en_US': 'Africa/Cairo', 'zh_Hans': '非洲/开罗', 'pt_BR': 'Africa/Cairo', 'ja_JP': 'Africa/Cairo'}}], 'type': 'select', 'human_description': {'en_US': 'Timezone', 'zh_Hans': '时区', 'pt_BR': 'Timezone', 'ja_JP': 'Timezone'}, 'form': 'form', 'llm_description': None}], 'description': {'human': {'en_US': 'A tool for getting the current time.', 'zh_Hans': '一个用于获取当前时间的工具。', 'pt_BR': 'A tool for getting the current time.', 'ja_JP': 'A tool for getting the current time.'}, 'llm': 'A tool for getting the current time.'}, 'output_schema': None, 'has_runtime_parameters': False, 'runtime_parameters': {'format': '%Y-%m-%d %H:%M:%S', 'timezone': 'UTC'}, 'provider_type': 'builtin'}, {'identity': {'author': 'LJX', 'name': 'biying_web_search', 'label': {'en_US': 'BingWebSearch', 'zh_Hans': '必应中国搜索', 'pt_BR': 'BingWebSearch', 'ja_JP': 'BingWebSearch'}, 'provider': 'biying', 'icon': None}, 'parameters': [{'name': 'query', 'label': {'en_US': 'Query string', 'zh_Hans': '查询语句', 'pt_BR': 'Query string', 'ja_JP': 'Query string'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': True, 'default': None, 'min': None, 'max': None, 'precision': None, 'options': [], 'type': 'string', 'human_description': {'en_US': 'used for searching', 'zh_Hans': '用于搜索网页内容', 'pt_BR': 'used for searching', 'ja_JP': 'used for searching'}, 'form': 'llm', 'llm_description': 'key words for searching'}, {'name': 'num_results', 'label': {'en_US': 'Number of results', 'zh_Hans': '搜索返回URL数量', 'pt_BR': 'Number of results', 'ja_JP': 'Number of results'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': True, 'default': None, 'min': 1, 'max': 40, 'precision': None, 'options': [], 'type': 'number', 'human_description': {'en_US': 'used for selecting the number of results', 'zh_Hans': '用于选择搜索返回的URL数量', 'pt_BR': 'used for selecting the number of results', 'ja_JP': 'used for selecting the number of results'}, 'form': 'llm', 'llm_description': None}], 'description': {'human': {'en_US': 'A tool for performing a Bing SERP search and extracting snippets and webpages.Input should be a search query.', 'zh_Hans': '一个用于执行 Bing SERP 搜索并提取片段和网页的工具。输入应该是一个搜索查询。输出是一个dict, 包含一个key：urls, value是一个list, 包含搜索结果的url。', 'pt_BR': 'A tool for performing a Bing SERP search and extracting snippets and webpages.Input should be a search query.', 'ja_JP': 'A tool for performing a Bing SERP search and extracting snippets and webpages.Input should be a search query.'}, 'llm': 'A tool for performing a Bing SERP search and extracting snippets and webpages.Input should be a search query.'}, 'output_schema': None, 'has_runtime_parameters': False, 'runtime_parameters': {}, 'provider_type': 'builtin'}, {'identity': {'author': 'Dify', 'name': 'simple_code', 'label': {'en_US': 'Code Interpreter', 'zh_Hans': '代码解释器', 'pt_BR': 'Interpretador de Código', 'ja_JP': 'Code Interpreter'}, 'provider': 'code', 'icon': None}, 'parameters': [{'name': 'language', 'label': {'en_US': 'Language', 'zh_Hans': '语言', 'pt_BR': 'Idioma', 'ja_JP': 'Language'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': True, 'default': None, 'min': None, 'max': None, 'precision': None, 'options': [{'value': 'python3', 'label': {'en_US': 'Python3', 'zh_Hans': 'Python3', 'pt_BR': 'Python3', 'ja_JP': 'Python3'}}, {'value': 'javascript', 'label': {'en_US': 'JavaScript', 'zh_Hans': 'JavaScript', 'pt_BR': 'JavaScript', 'ja_JP': 'JavaScript'}}], 'type': 'string', 'human_description': {'en_US': 'The programming language of the code', 'zh_Hans': '代码的编程语言', 'pt_BR': 'A linguagem de programação do código', 'ja_JP': 'The programming language of the code'}, 'form': 'llm', 'llm_description': 'language of the code, only "python3" and "javascript" are supported'}, {'name': 'code', 'label': {'en_US': 'Code', 'zh_Hans': '代码', 'pt_BR': 'Código', 'ja_JP': 'Code'}, 'placeholder': None, 'scope': None, 'auto_generate': None, 'template': None, 'required': True, 'default': None, 'min': None, 'max': None, 'precision': None, 'options': [], 'type': 'string', 'human_description': {'en_US': 'The code to be executed', 'zh_Hans': '要执行的代码', 'pt_BR': 'O código a ser executado', 'ja_JP': 'The code to be executed'}, 'form': 'llm', 'llm_description': 'code to be executed, only native packages are allowed, network/IO operations are disabled.'}], 'description': {'human': {'en_US': "Run code and get the result back. When you're using a lower quality model, please make sure there are some tips help LLM to understand how to write the code.", 'zh_Hans': '运行一段代码并返回结果。当您使用较低质量的模型时，请确保有一些提示帮助 LLM 理解如何编写代码。', 'pt_BR': 'Execute um trecho de código e obtenha o resultado de volta. quando você estiver usando um modelo de qualidade inferior, certifique-se de que existam algumas dicas para ajudar o LLM a entender como escrever o código.', 'ja_JP': "Run code and get the result back. When you're using a lower quality model, please make sure there are some tips help LLM to understand how to write the code."}, 'llm': 'A tool for running code and getting the result back. Only native packages are allowed, network/IO operations are disabled. and you must use print() or console.log() to output the result or result will be empty.'}, 'output_schema': None, 'has_runtime_parameters': False, 'runtime_parameters': {}, 'provider_type': 'builtin'}],
            'instruction': '请选择合适的工具(Function)回答用户的问题',
            'query': '现在的时间是？',
            'maximum_iterations': 3
        }
    )
    for x in factory.invoke_agent_strategy(req):
        print(x)
