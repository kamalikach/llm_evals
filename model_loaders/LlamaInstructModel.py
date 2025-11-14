from .BaseModel import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
#import torch 

class LlamaInstructModel(BaseModel):
    def load(self, model_args=None):
        print(self.model_id)
        model_name = self.model_id

        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",        # same as your original dtype
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer, self.llm

        #print(self.model_id)
        #model_name = self.model_id
        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        #model = AutoModelForCausalLM.from_pretrained(model_name, 
        #        dtype=torch.bfloat16, device_map="auto")
        
        #self.model = model
        #self.tokenizer = tokenizer
        #self.model.config.pad_token_id = self.model.config.eos_token_id
        #return tokenizer, model

    #def prompt_response(self, system_prompt, user_prompt):
    #    prompt = self.tokenizer.apply_chat_template( [{"role": "user", 
    #        "content": system_prompt + ':' + user_prompt}],tokenize=False, add_generation_prompt=True)
    #    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    #    outputs = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.model.config.eos_token_id)
    #    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #   cleaned_response = response.split("assistant", 1)[1].strip()
    #   return cleaned_response

    def prompt_response(self, system_prompt, user_prompt):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}: {user_prompt}"}],
            tokenize=False,
            add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None,
        )

        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        #if "assistant" in response_text:
        #    response_text = response_text.split("assistant", 1)[1].strip()

        return response_text
