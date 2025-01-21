
def double_star(dialogue_history):
    dialogue_history[-1] = f"** {dialogue_history[-1]}"
    return dialogue_history

def default(input):
    return input