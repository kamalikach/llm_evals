from transformers import AutoModelForCausalLM, AutoTokenizer
from .BaseModel import BaseModel
import torch 

class LlamaInstructModel(BaseModel):
    def load(self, model_args=None):
        print(self.model_id)
        model_name = self.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                dtype=torch.bfloat16, device_map="auto")
        
        self.model = model
        self.tokenizer = tokenizer
        self.model.config.pad_token_id = self.model.config.eos_token_id
        return tokenizer, model

    def prompt_response(self, system_prompt, user_prompt):
        prompt = self.tokenizer.apply_chat_template( [{"role": "user", 
            "content": system_prompt + ':' + user_prompt}],tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=512, pad_token_id=self.model.config.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('Response:', response)

        #cleaned_response = response[len(prompt):].strip()
        cleaned_response = response.split("assistant", 1)[1].strip()
        print('Cleaned response:', cleaned_response)
        return cleaned_response

