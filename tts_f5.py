
import os
import csv
from tqdm import tqdm
import subprocess
import shutil
import sys
from gtts import gTTS

import re
import inflect
from decimal import Decimal
import spacy
nlp = spacy.load("en_core_web_sm")

def convert_dollar_to_words(amount):
    # Create an inflect engine
    p = inflect.engine()
    # Ensure the amount is a Decimal for precise handling of money
    amount = amount.replace(",", "").strip('.')
    decimal_amount = Decimal(amount)
    # Separate the dollar and cent parts
    dollars = int(decimal_amount)
    cents = int((decimal_amount - dollars) * 100)
    # Convert dollars and cents to words
    dollar_word = p.number_to_words(dollars, andword='')
    cent_word = p.number_to_words(cents, andword='')
    # Format the final string
    if dollars == 0 and cents > 0:
        cent_part = "one cent" if cents == 1 else f"{cent_word} cents"
        return cent_part
    elif cents == 0:
        dollar_part = "one dollar" if dollars == 1 else f"{dollar_word} dollars"
        return dollar_part
    else:
        dollar_part = "one dollar" if dollars == 1 else f"{dollar_word} dollars"
        cent_part = "and one cent" if cents == 1 else f"and {cent_word} cents"
        return f"{dollar_part} {cent_part}"

replacements = {
    'GB': ' gigabytes',
    'BOGO': 'buy-one-get-one',
    '5-7': 'five to seven',
    '4-6': 'four to six',
    'bryanthecoolestguy': 'bryan the coolest guy',
    '@gmail.com': ' at G mail dot com',
    'COPD': 'C O P D',
    '12%': 'twelve percent',
    '100%': 'one hundred percent',
    'ADHD': 'A D H D',
    'SLR': 'S L R',
    'ISO': 'I S O',
    'LCD': 'L C D',
    'HD': 'H D',
    '1080p': 'ten eighty P',
    'BPA-free': 'B P A free',
    '4-2=': 'four minus two equals',
    'fr fr': 'for real for real',
    'APR': 'A P R',
    '100 all the way up to 12800': 'one hundred all the way up to twelve thousand eight hundred',
    '100 to 12800':  'one hundred to twelve thousand eight hundred',
    '100 all da way up to 25600': 'one hundred all da way up to twenty five thousand six hundred',
    '100 all the way up to 25600': 'one hundred all the way up to twenty five thousand six hundred',
    '100 to 25,600': 'one hundred to twenty five thousand six hundred',
    '100 to 12,800': 'one hundred to twelve thousand eight hundred',
    "= -1/x^2": "equals negative one over X squared",
    "f(x) =": "F of X equals",
    "f(x)": "F of X",
    "1/x - 0/0": "one over X minus zero over zero",
    "1/x": "one over X",
    "f'(x) =": "F prime of X equals",
    "0/0": 'zero over zero',
    "x=0": "X equals zero",
    '60s': 'sixties',
    '184792': 'one eight four seven nine two',
    '123 Main Street, Apartment 4B': 'one two three Main Street, Apartment four B',
    '1234567890': 'one two three four five six seven eight nine zero',
    '100 folks in marketing, 200 in sales, 300 in customer service, and 400 in accounting': 'one hundred folks in marketing, two hundred in sales, three hundred in customer service, and four hundred in accounting',
    '100 people in marketing, 200 in sales, 300 in customer service, and about 400 in accounting': 'one hundred people in marketing, two hundred in sales, three hundred in customer service, and four hundred in accounting',
    '100 folks in marketing, 200 in sales, 300 up in customer service, and 400 working in accounting': 'one hundred folks in marketing, two hundred in sales, three hundred up in customer service, and four hundred working in accounting',
    'BMWs': 'bee em double youse',
    '320i': 'three twenty eye',
    '328i': 'three twenty eight eye',
    '335i': 'three thirty five eye',
    '340i': 'three forty eye',
    'AA1234': 'A A one two three four',
    'A1C': 'A one C',
    '90210': 'nine zero two one zero',
}

def correct(line):
    # Regex pattern to find dollar amounts
    if '$' in line:
        # dollar_regex = r"\$(\d{1,3}(,\d{3})*(\.\d{1,2})?)"
        dollar_regex = r"\$([\d,\.]+)"
        line = re.sub(dollar_regex, lambda m: convert_dollar_to_words(m.group(1)), line)
    for phrase, new_phrase in replacements.items():
        if phrase in line:
            line = line.replace(phrase, new_phrase) 
    if '*' in line:
        line = re.sub(r'\*[^*]*\*', '', line) # remove text in * such as *looking through paper* 
    line = re.sub(r'\b[mM]*[-]?h+m*\b', 'Yeah', line) # replace all variants of mhmm with yeah 
    line = re.sub(r'\b[mM]{3,}\b', 'Yeah', line) # replace all variants of mmm with yeah 
    if line.strip() == 'User: 2':
        line = 'User: Well, okay. I know that four minus two equals two.'
    return line
    

def preprocess(line):
    line = line.replace('"', '')
    return line

def chunkify(line, min_words):
    doc = nlp(line)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    temp_chunk = ""
    for sentence in sentences:
        # Add the current sentence to the temporary chunk
        if temp_chunk:
            temp_chunk += " " + sentence
        else:
            temp_chunk = sentence

        # Check the word count
        if len(temp_chunk.split()) >= min_words:
            # If the word count is sufficient, add to chunks and reset temp_chunk
            chunks.append(temp_chunk.strip())
            temp_chunk = ""

    # Handle leftover chunk
    if temp_chunk:
        # If there are existing chunks and the leftover chunk has fewer than min_words,
        # merge it with the last chunk
        if chunks and len(temp_chunk.split()) < min_words:
            chunks[-1] += " " + temp_chunk.strip()
        else:
            chunks.append(temp_chunk.strip())

    return chunks

orig_aave_audio = "audio_reference/ATL_se0_ag2_f_02_04sec.wav"
orig_aave_transcript = "So are there particular topics that we have to talk about on this interview?"

user_sae_audio = "audio_reference/1926-147987-0005-90pitch-05sec.wav"
user_sae_transcript = "Maybe Jim would be willing to go over there and sleep. And you could come here nights"

base_sae_audio = "audio_reference/298-126790-0034-05sec.wav"
base_sae_transcript = "For my part, I am clear for doctoring, though Harvey says I am killing myself with medicines"



aave_model_to_voice = [('c', '1'), ('c', '2'), ('c', '3'), ('g', '0')]
sae_model_to_voice = [('g', '0')]

output_dir = "audio_generations/f5tts/outputs_turnsplit"
output_hyperparam_dir = f"{output_dir}/{os.path.basename(orig_aave_audio)[:-4]}_defaultconfig_withremovesilence"
os.makedirs(output_hyperparam_dir, exist_ok=True)
one_version_done = {} # to avoid running user audio generation for different variants of same dialogue

voice_conditions = [('aave', aave_model_to_voice), ('sae', sae_model_to_voice), ('control', sae_model_to_voice)]


# collect all unique dialogue variants first
files = [
    'data/text_chatbot/100_selected_claude-3-5-sonnet-20241022-significant.csv',
    'data/text_chatbot/100_selected_claude-3-5-sonnet-20241022.csv',
    'data/text_chatbot/100_selected_gpt-4o-2024-11-20-saecasual.csv'
]
dialogue_data = {}
for file in files:
    reader = csv.DictReader(open(file))
    for data in reader:
        if 'persona-aave-significant' in data:
            output = data['persona-aave-significant']
            dialogue_data[output] = {
                'role': data['dialogue_id'].split('-')[0],
                'dialogue': data['dialogue_id'].split('-')[1],
                'output': output,
                'model': 'c', # claude
                'prompt': '3' # signficant
            }
        if 'persona-aave-mid' in data:
            output = data['persona-aave-mid']
            dialogue_data[output] = {
                'role': data['dialogue_id'].split('-')[0],
                'dialogue': data['dialogue_id'].split('-')[1],
                'output': output,
                'model': 'c', # claude
                'prompt': '2' # mid
            }
        if 'persona-aave-slight' in data:
            output = data['persona-aave-slight']
            dialogue_data[output] = {
                'role': data['dialogue_id'].split('-')[0],
                'dialogue': data['dialogue_id'].split('-')[1],
                'output': output,
                'model': 'c', # claude
                'prompt': '1' # slight
            }
        if 'persona-sae-casual' in data:
            output = data['persona-sae-casual']
            dialogue_data[output] = {
                'role': data['dialogue_id'].split('-')[0],
                'dialogue': data['dialogue_id'].split('-')[1],
                'output': output,
                'model': 'g', # gpt
                'prompt': '0' # SAE
            }

# to support running parallel processes for more efficient audio generation
dialogue_data_items = sorted(dialogue_data.items())
start, end = 0, len(dialogue_data_items)
if len(sys.argv) > 1:
    start, end = int(sys.argv[1]), int(sys.argv[2])

print(f'Generating audio for dialogues [{start},{end})')
print(len(dialogue_data_items[start:end]))

# iterate over all collected dialogue variants
for didx, (dialogue, data) in enumerate(dialogue_data_items[start:end]):
    print(didx, f"{data['role']}-{data['dialogue']}-{data['model']}-{data['prompt']}")
    for version, identifiers in voice_conditions:
        chatbot_audio_fn, chatbot_ref_text, audio_fn, ref_text = None, None, None, None
        if (data['model'], data['prompt']) in identifiers:
            if version == 'aave':
                chatbot_audio_fn = orig_aave_audio
                chatbot_ref_text = orig_aave_transcript
            elif version == 'sae':
                chatbot_audio_fn = base_sae_audio
                chatbot_ref_text = base_sae_transcript
            elif version == 'control':
                chatbot_audio_fn = gTTS
            if chatbot_audio_fn is None:
                continue
            dialogue_audio_save_dir = f"{output_hyperparam_dir}/{data['role']}-{data['dialogue']}-{data['model']}-{data['prompt']}-{version}"
            if not os.path.exists(dialogue_audio_save_dir):
                os.makedirs(dialogue_audio_save_dir, exist_ok=True)
                with open(f"{dialogue_audio_save_dir}/transcript.txt", "w") as w:
                    w.write(data['output'])
                lines = data['output'].split('\n')
                for i, orig_line in enumerate(tqdm(lines, desc="TTS")):
                    audio_fn, ref_text = None, None
                    line = correct(orig_line)
                    line = preprocess(line)
                    if line.startswith('System: '):
                        line = line.replace('System: ', '')
                        speaker = 'chatbot'
                        audio_fn = chatbot_audio_fn
                        ref_text = chatbot_ref_text
                    elif line.startswith('User: '):
                        if f"{data['role']}-{data['dialogue']}" not in one_version_done:
                            line = line.replace('User: ', '')
                            speaker = 'user'
                            audio_fn = user_sae_audio
                            ref_text = user_sae_transcript
                        else:
                            print('Skipping User because already done or rerunning SAE/control chatbot generation only')
                    else:
                        raise Exception(f'Line `{line}` does not have a speaker tag!')
                    if audio_fn is not None:
                        # split turn into sentences for better audio generation
                        sentences = chunkify(line, min_words=6)
                        for j, sentence in enumerate(sentences):
                            # print(sentence)
                            save_filename = f"turn_{i+1}_s{j+1}_{speaker}.mp3"
                            if version in {'aave', 'sae'}:
                                command =f"""
                                f5-tts_infer-cli \
                                --model "F5-TTS" \
                                --ref_audio "{audio_fn}" \
                                --ref_text "{ref_text}" \
                                --gen_text "{sentence}" \
                                --output_dir "{dialogue_audio_save_dir}" \
                                --output_file "{save_filename}" \
                                --remove_silence 
                                """.strip()
                                speed80_change_command = f'ffmpeg -y -i "{dialogue_audio_save_dir}/{save_filename}" -filter:a "atempo=0.80" {dialogue_audio_save_dir}/{save_filename[:-4]}_80speed.mp3'
                                speed85_change_command = f'ffmpeg -y -i "{dialogue_audio_save_dir}/{save_filename}" -filter:a "atempo=0.85" {dialogue_audio_save_dir}/{save_filename[:-4]}_85speed.mp3'
                                speed90_change_command = f'ffmpeg -y -i "{dialogue_audio_save_dir}/{save_filename}" -filter:a "atempo=0.90" {dialogue_audio_save_dir}/{save_filename[:-4]}_90speed.mp3'
                                try:
                                    result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    result = subprocess.run(speed90_change_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    result = subprocess.run(speed85_change_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    result = subprocess.run(speed80_change_command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                except subprocess.CalledProcessError as e:
                                    print(f"Error: Command failed with error code {e.returncode}")
                                    print("Error Output:", e.stderr.decode())
                            else:
                                tts = audio_fn(sentence)
                                tts.save(f"{dialogue_audio_save_dir}/{save_filename}")
                
                # case: previous variant of dialogue already processed, need to copy user audio from that directory
                if f"{data['role']}-{data['dialogue']}" in one_version_done:
                    previous_identifier = one_version_done[f"{data['role']}-{data['dialogue']}"]
                    for root, _, files in os.walk(f"{output_hyperparam_dir}/{previous_identifier}"):
                            for file in files:
                                if "user" in file:  # Check if user utterance audio file
                                    source_file = os.path.join(root, file)
                                    destination_file = os.path.join(dialogue_audio_save_dir, file)
                                    shutil.copy2(source_file, destination_file)
                # case: no previous variant of dialogue already processed, this was the first processing, so need to record this
                if f"{data['role']}-{data['dialogue']}" not in one_version_done:
                    one_version_done[f"{data['role']}-{data['dialogue']}"] = f"{data['role']}-{data['dialogue']}-{data['model']}-{data['prompt']}-{version}"
            

                                                    

                    


