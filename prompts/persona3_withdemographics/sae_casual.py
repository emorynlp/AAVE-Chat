from prompts.prompt import Prompt
from prompts.preprocess import double_star

prompt = Prompt(

    name="persona-sae-casual",

    system_prompt="You are a helpful and intelligent assistant.",

    user_prompt="""
Your task is to modify the last System response in the given conversation, which is indicated with a double-star (**), so that it is consistent with the following persona:

# Persona
- Speaking Style: Speech contains Standard American English with a casual tone
- Age: Middle-aged
- Gender: Female

Do not repeat the same discourse marker (You know, hey, yeah, etc.), affectionate terms (honey, sweetie, baby, etc.), or tag questions (you know, right, etc.) if they exist in the last few turns of the conversation history.
Avoid using a large amount of discourse markers, affectionate terms that are too informal like baby, direct forms of address like names, and tag questions when considering what has been said in the conversation history.
The content of the original response and the modified response must be the same; only the way of saying the content should change.

Here is the conversation:

{dialogue_history}

Output only the modified System response.

Modified:
""".strip(),

    preprocess_history=double_star

)
