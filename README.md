# AI_Subtitle_Translator (RO-EN and EN-RO)

This script translates subtitles from Romanian to English and from English to Romanian by using the M2M100 AI model.
The script reads subtitles from a `.srt` file, translates each subtitle line, and writes the translated subtitles to a new `.srt` file.

## Requirements:

- `transformers`
- `sentencepiece`
- `torch==2.4.1+cu124`
- `torchvision==0.19.1+cu124`
- `torchaudio==2.4.1+cu124`
