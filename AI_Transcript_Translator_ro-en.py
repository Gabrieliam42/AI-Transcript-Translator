# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import os

# Define the model name for M2M100 (1.2 billion parameters)
model_name = "facebook/m2m100_1.2B"
cache_dir = os.getcwd()  # Set cache directory to current working directory

# Load the tokenizer and model with specified cache directory
tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

# Ensure the model runs on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate_to_english(text):
    tokenizer.src_lang = "ro"
    encoded_text = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

# Read from a .srt file
with open('subtitle.srt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Translate each subtitle line (excluding time codes and blank lines)
translated_lines = []
for line in lines:
    if line.strip() and not line.strip().isdigit() and '-->' not in line:
        translated_line = translate_to_english(line.strip())
        translated_lines.append(translated_line)
    else:
        translated_lines.append(line.strip())

# Write the translated subtitles to a new file
with open('translated_subtitle.srt', 'w', encoding='utf-8') as file:
    file.write("\n".join(translated_lines))

print("Translated subtitles have been saved to translated_subtitle.srt")
