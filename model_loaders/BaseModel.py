from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_id):
        self.model_id = model_id

    @abstractmethod
    def load(self, model_args=None):
        pass

    @abstractmethod
    def prompt_response(self, system_prompt, user_prompt):
        pass


