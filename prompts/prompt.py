from dataclasses import dataclass, field

@dataclass
class Prompt:
    name: str
    system_prompt: str
    user_prompt: str
    preprocess_history: callable

    def __post_init__(self):
        self.system_prompt = self.system_prompt.strip()
        self.user_prompt = self.user_prompt.strip()