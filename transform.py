# Set the random seeds for reproducibility
seed = 542
import random
import torch
import numpy as np
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

import os
import gc
import json
import pandas as pd
import transformers
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts.prompt import Prompt
import prompts.persona3_withdemographics

################################################################################################################


def chatbot_turn(client, model_type: str, dialogue_history, prompt: Prompt=None):
    if prompt is not None:
        dialogue_history = prompt.preprocess_history(dialogue_history)
        user_prompt = prompt.user_prompt.format(dialogue_history='\n'.join(dialogue_history))
    if 'gpt' in model_type:
        response = client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.0
        )
        assistant_response = response.choices[0].message.content.strip().replace("‘", "'").replace("’", "'")
        if not assistant_response.startswith('System'):
            assistant_response = f"System: {assistant_response}"
    elif 'claude' in model_type:
        assistant_response = ""
        while not assistant_response:
            try:
                message = client.messages.create(
                    model=model_type,
                    max_tokens=500,
                    temperature=0.0,
                    system=prompt.system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                assistant_response = message.content[0].text.strip().replace("‘", "'").replace("’", "'")
                if not assistant_response.startswith('System'):
                    assistant_response = f"System: {assistant_response}"
            except anthropic.InternalServerError as e:
                print(e)
    elif 'neuralmagic' in model_type:
        sampling_params = SamplingParams(
            n=1,
            use_beam_search=True, 
            best_of=5,
            max_tokens=500,
            repetition_penalty=1.0,
            early_stopping=True,
            temperature=0,
        )
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompts = client[1].apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        outputs = client[0].generate(prompts, sampling_params)
        assistant_response = outputs[0].outputs[0].text
        if not assistant_response.startswith('System'):
            assistant_response = f"System: {assistant_response}"
    dialogue_history[-1] = assistant_response
    return assistant_response, dialogue_history


if __name__ == "__main__":

    testing = [
        ("neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16", prompts.persona3_withdemographics.aave_slight.prompt),
        ("neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16", prompts.persona3_withdemographics.aave_mid.prompt),
        ("neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16", prompts.persona3_withdemographics.aave_significant.prompt),
        ("gpt-4o-2024-11-20", prompts.persona3_withdemographics.sae_casual.prompt),
        ("gpt-4o-2024-11-20", prompts.persona3_withdemographics.aave_slight.prompt),
        ("gpt-4o-2024-11-20", prompts.persona3_withdemographics.aave_mid.prompt),
        ("gpt-4o-2024-11-20", prompts.persona3_withdemographics.aave_significant.prompt),
        ("claude-3-5-sonnet-20241022", prompts.persona3_withdemographics.aave_slight.prompt),
        ("claude-3-5-sonnet-20241022", prompts.persona3_withdemographics.aave_mid.prompt),
        ("claude-3-5-sonnet-20241022", prompts.persona3_withdemographics.aave_significant.prompt),
    ]

    mapping = {
        'slight': prompts.persona3_withdemographics.aave_slight.prompt,
        'mid': prompts.persona3_withdemographics.aave_mid.prompt,
        'significant': prompts.persona3_withdemographics.aave_significant.prompt
    }

    eightbit = False
    fourbit = False
    quant = ''
    savefilesuffix = ''
    if len(sys.argv) > 1:
        testing.append((sys.argv[1], mapping[sys.argv[2]]))
        savefilesuffix = sys.argv[2]
        eightbit = sys.argv[3] == '8bit'
        fourbit = sys.argv[3] == '4bit'
        quant = sys.argv[3]

    print(testing)
    print('eightbit', eightbit)
    print('fourbit', fourbit)

    files = ['soda/100_selected']

    for file in files:
        df = pd.read_csv(open(f'data/{file}.csv'))
        for model_type, test_prompt in testing:
            if 'gpt' in model_type:
                # GPT
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            elif 'claude' in model_type:
                # ANTHROPIC
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            elif 'neuralmagic' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_type)
                client = [LLM(model=model_type, tensor_parallel_size=1, max_model_len=2048), tokenizer]

            if test_prompt:
                name = test_prompt.name
            else:
                name = model_type
            print()
            print('#'*50)
            print(name)
            print('#'*50)
            print()
            transformed_dialogues = [None] * len(df)
            for i, conversation in enumerate(df['Conversation']):
                if True: #i < 5:
                    if name not in df.columns:
                        current_entry = None
                    else:
                        current_entry = df.at[i, name]
                    if pd.notna(current_entry):
                        print(f"Skipping transformation for conversation {i} as '{name}' already exists.")
                        transformed_dialogues[i] = current_entry
                        continue
                    dialogue_history = []
                    turns = conversation.split('\n')
                    for turn in turns:
                        dialogue_history += [turn]
                        if turn.startswith("System:"):
                            transformed_response, dialogue_history = chatbot_turn(client, model_type, dialogue_history, test_prompt)
                        elif turn.startswith("User:"):
                            ...
                        else:
                            raise Exception(f"Turn `{turn}` does not start with `User:` or `System:`")
                    transformed_convo = '\n'.join(dialogue_history)
                    print(f'## {i} ##')
                    print(transformed_convo)
                    print()
                    transformed_dialogues[i] = transformed_convo
            df[name] = transformed_dialogues

            if 'neuralmagic' in model_type:
                del client[0]
                gc.collect()
                torch.cuda.empty_cache()

        df.to_csv(f'data/results/{file.split("/")[-1]}_{model_type.replace("/", "--")}.csv', index=False)