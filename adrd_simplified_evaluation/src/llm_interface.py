from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

class LLMWrapper:

    def __init__(self, config):

        self.llm = LLM(**config.LLM)

        self.sampling_params = SamplingParams(**config.sampling_params)

        if config.LLM.enable_lora:
            self.lora_request = LoRARequest("adapter", 1, config.LLM.lora_path)
        else:
            self.lora_request = None

    def generate(self, messages, enable_thinking=None):
        # Generate completions from a set of messages using vllm.LLM.chat.
        #
        # `messages` should be a list of dictionaries. This function uses
        # vllm.LLM.chat, which applies the chat template and tokenizes the
        # messages. Usually only instruction-tuned models have a chat template,
        # base models may perform poorly.

        print("Processing prompts... ")

        if enable_thinking is not None:
            completions = self.llm.chat(
                messages=messages, 
                sampling_params=self.sampling_params,
                lora_request=self.lora_request,
                chat_template_kwargs={"enable_thinking": enable_thinking},  
            )
        else:
            completions = self.llm.chat(
                messages=messages, 
                sampling_params=self.sampling_params,
                lora_request=self.lora_request
            )

        return completions
    
