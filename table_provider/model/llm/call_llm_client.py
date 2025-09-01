# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

import tiktoken  # for counting tokens
from tenacity import retry, stop_after_attempt, wait_random_exponential

from table_provider import logger
from table_provider.config.entity.azure_openai_config import AzureOpenAIConfig, DEFAULT_CALL_LLM_CONFIG_PATH, \
    AzureOpenAIClientConfig
from table_provider.prompt.call_llm_prompt import CallLLMPrompt, CallLLMPromptType


class CallLLMClient:
    """Class for calling the OpenAI Language Model API."""

    def __init__(self, config: AzureOpenAIConfig = None):
        if config is None:
            with open(DEFAULT_CALL_LLM_CONFIG_PATH) as config_file:
                json_config = json.load(config_file)
            self.config = AzureOpenAIConfig(json_config["call_llm_config"]["endpoint"],
                                       json_config["call_llm_config"]["deployment_name"],
                                       json_config["call_llm_config"]["model_name"])
            self.model_name = json_config["call_llm_config"]["model_name"]
        else:
            self.config = config
            self.model_name = config.model_name
        self.client_config = AzureOpenAIClientConfig(self.config)
        self.client = self.client_config.client
        self.logger = logger

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(text))

    def num_tokens_list(self, text):
        """Return the number of tokens in a list of strings."""
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode("".join([str(item) for item in text])))

    def truncated_string(
        self, string: str, token_limit: int, print_warning: bool = True
    ) -> str:
        """Truncate a string to a maximum number of tokens."""
        encoding = tiktoken.encoding_for_model(self.model_name)
        encoded_string = encoding.encode(string)
        truncated_string = encoding.decode(encoded_string[:token_limit])
        if print_warning and len(encoded_string) > token_limit:
            logger.warn(
                f"Warning: Truncated string from {len(encoded_string)} tokens to {token_limit} tokens."
            )
        return truncated_string


    def parse_text_into_table(self, string: str) -> str:
        prompt = CallLLMPrompt.get_prompt(CallLLMPromptType.PARSE_TEXT_INTO_TABLE, string = string)
        return self.generate_text(prompt)

    def fill_in_cell_with_context(self, context: str) -> str:
        """Fill in the blank based on the column context."""
        prompt = CallLLMPrompt.get_prompt(CallLLMPromptType.FILL_IN_CELL_WITH_CONTEXT, context = context)
        return self.generate_text(prompt)

    def fill_in_column_with_context(self, context: str) -> str:
        """Fill in the blank based on the column cells context."""
        prompt = CallLLMPrompt.get_prompt(CallLLMPromptType.FILL_IN_COLUMN_WITH_CONTEXT, context = context)
        return self.generate_text(prompt)

    def call_llm_summarization(self, context: str) -> str:
        """Summarize the table context to a single sentence."""
        prompt = CallLLMPrompt.get_prompt(CallLLMPromptType.CALL_LLM_SUMMARIZATION, context = context)
        return self.generate_text(prompt)

    def call_llm_code_generation(self, context: str) -> str:
        """Synthesize code snippet from the table context."""
        prompt = CallLLMPrompt.get_prompt(CallLLMPromptType.CALL_LLM_CODE_GENERATION, context = context)
        return self.generate_text(prompt)

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text(self, prompt: str) -> str:
        """Generate text based on the prompt and instruction."""
        converted_prompt = {"role": "user", "content": prompt}
        completion = self.client.chat.completions.create(
            model=self.config.deployment_name,
            messages=[converted_prompt],
            max_tokens=8192,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        return json.loads(completion.to_json())["choices"][0]["message"]["content"].strip()

    @retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(1000))
    def generate_text_with_messages(self, messages):
        """Generate text based on the prompt and instruction."""
        response = self.client.chat.completions.create(
            model=self.config.deployment_name,
            messages=messages,
            max_tokens=3000,
            temperature=0,
        )

        output = json.loads(response.model_dump_json())
        data = output["choices"][0]['message']['content']
        return data
