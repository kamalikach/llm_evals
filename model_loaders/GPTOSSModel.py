from .BaseModel import BaseModel
from vllm import LLM, SamplingParams,TokensPrompt
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

class GPTOSSModel(BaseModel):
    def load(self, model_args=None):
        print(self.model_id)

        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",       
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9
        )

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self.sampling_params = SamplingParams(
                temperature=0.0, max_tokens=4096,stop_token_ids=stop_token_ids)
        return 

    def prompt_response(self, system_prompt, user_prompt):
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(system_prompt),),
                Message.from_role_and_content(Role.USER, user_prompt),
            ])
        prefill_ids = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        outputs = self.llm.generate(prompts=[TokensPrompt(prompt_token_ids=prefill_ids)], sampling_params=self.sampling_params)

        output_tokens = outputs[0].outputs[0].token_ids
        entries = self.encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

        try:
            final_messages = [msg for msg in entries if msg.channel == 'final']
            if final_messages:
                response_text = final_messages[-1].content[0].text
        
            if entries and entries[-1].content:
                response_text = entries[-1].content[0].text
        
        except (IndexError, AttributeError) as e:
            print(f"Error extracting response: {e}")
            print(f"Entries: {entries}")
            response_text = ""
        
        print('Model response:', response_text)

        return response_text

