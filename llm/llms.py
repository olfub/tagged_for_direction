from pathlib import Path

from anthropic import Anthropic
import torch
#transformers=None
#import transformers
#from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


class LLMInterface:

    def __init__(self, model_name, api_key=None):
        self.model_name = model_name

    def query(self, messages, dry_run=False, **kwargs):
        if dry_run:
            print(messages)
            return None

        return self.do_query(messages, **kwargs)

    def do_query(self, messages, **kwargs) -> str:
        raise NotImplemented()


class AntrophicInterface(LLMInterface):

    def __init__(self, model_name, api_key=None):
        super().__init__(model_name, api_key)
        self.client = Anthropic(
            api_key=api_key,
        )

    def do_query(self, messages, temperature=0.0, max_tokens=4096):
        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = None

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        return "".join([cb.text for cb in message.content])


class TransformersInterface(LLMInterface):

    def __init__(self, model_name, api_key=None, model_id=None):
        super().__init__(model_name, api_key)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def do_query(self, messages, temperature=0.0, max_tokens=4096):
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=None if temperature == 0.0 else temperature,
            do_sample=(temperature == 0.0)
        )
        return outputs[0]["generated_text"][-1]["content"]


class QwenInterface(LLMInterface):

    def __init__(self, model_name, api_key=None, model_id=None):
        super().__init__(model_name, api_key)
        print("loading", model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def do_query(self, messages, temperature=0.0, max_tokens=4096):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=None if temperature == 0.0 else temperature,
            do_sample=(temperature == 0.0)
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        #return outputs[0]["generated_text"][-1]["content"]


class OpenAIInterface(LLMInterface):

    def __init__(self, model_name, api_key=None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=api_key,
        )

    def do_query(self, messages, temperature=0.0, max_tokens=4096):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content


llm_configs = {
    "claude-3-5-sonnet-20241022": {
        "hname": "Claude 3.5 Sonnet",
        "api": "anthropic",
        "interface": AntrophicInterface,
    },
    "Llama-3.3-70B-Instruct": {
        "hname": "Llama 3.3 70B Instruct",
        "api": None,
        "interface": TransformersInterface,
        "create_parameters": {
            "model_id": "meta-llama/Llama-3.3-70B-Instruct"
        }
    },
    "Qwen2.5-72B-Instruct": {
        "hname": "Qwen2.5 72B Instruct",
        "api": None,
        "interface": QwenInterface,
        "create_parameters": {
            "model_id": "Qwen/Qwen2.5-72B-Instruct"
        }
    },
    "gpt-4-0613": {
        "hname": "GPT-4 (0613)",
        "api": "openai",
        "interface": OpenAIInterface,
    },
    "gpt-4o-2024-08-06": {
        "hname": "GPT-4o (2024-08-06)",
        "api": "openai",
        "interface": OpenAIInterface,
    }
}


class LLMProvider:

    def __init__(self, key_dir=None):
        self._cache = {}
        self._key_dir: Path = key_dir

    def _create_interface(self, model_name) -> LLMInterface:
        llm_config = llm_configs[model_name]

        api_key = None
        if self._key_dir is not None:
            api_name = llm_config["api"]
            if api_name != "__local" and api_name is not None:
                key_file = self._key_dir / api_name
                if key_file.exists():
                    api_key = key_file.read_text()

        create_parameters = llm_config.get("create_parameters", {})
        interface = llm_config["interface"](model_name, api_key, **create_parameters)
        return interface

    def get_interface(self, name) -> LLMInterface:
        interface = self._cache.get(name, None)
        if interface is not None:
            return interface

        interface = self._create_interface(name)
        self._cache[name] = interface
        return interface
