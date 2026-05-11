# IoT-Voice-Command-Project
Evaluating masked voice command injection attacks on Google Home using TTS and human speech

## Project Overview

This project investigates whether perceptually masked voice commands can remain recognizable to automatic speech recognition (ASR) systems and smart assistants while becoming less intelligible to nearby listeners.

The experiments compare:

* Human-spoken commands
* Synthesized TTS commands

Under:

* Light masking
* Strong masking
* Hidden masking conditions

Evaluation was performed using:

* OpenAI Whisper
* Google Home playback experiments

## Repository Structure

```
IOT-VOICE-COMMAND-PROJECT/
│
├── data/
│   ├── google_home_results/
│   ├── processed/
│   ├── raw/
│   └── whisper_results/
│
├── graphs/
├── notebooks/
├── scripts/
│
├── .gitignore
├── README.md
└── requirements.txt

```

## Directory Overview

### data/

Stores audio datasets and evaluation results.

* **raw/**: original audio files
* **processed/**: normalized and processed WAV files
* **whisper_results/**: Whisper transcription outputs
* **google_home_results/**: Google Home trial results

### graphs/

Generated visualization outputs:

* ASR accuracy graphs
* Google Home success graphs
* Human vs TTS comparisons
* Hidden vs normal comparisons

### scripts/

Contains preprocessing, masking, evaluation, and analysis scripts.

## Python Scripts

### convert_mp3_to_wav.py

Converts ElevenLabs MP3 files into WAV format.
`python scripts/convert_mp3_to_wav.py`

### convert_human_m4a_to_wav.py

Converts human-recorded M4A files into WAV format.
`python scripts/convert_human_m4a_to_wav.py`

### normalize_audio.py

Normalizes audio loudness across the dataset.
`python scripts/normalize_audio.py`

### generate_hidden_commands.py

Generates perceptually masked hidden-command audio using:

* lowpass filtering
* reverberation
* gain reduction
* rain embedding
`python scripts/generate_hidden_commands.py`

### run_whisper.py

Runs OpenAI Whisper transcription on generated samples.
`python scripts/run_whisper.py`

### analyze_labeled_results.py

Analyzes labeled Whisper transcription results.
`python scripts/analyze_labeled_results.py`

### analyze_labeled_results2.py

Generates grouped ASR comparison graphs.
`python scripts/analyze_labeled_results2.py`

### analyze_google_home_results.py

Analyzes Google Home wake-word and execution success rates.
`python scripts/analyze_google_home_results.py`

## Experimental Workflow

1. **Convert Audio**
`python scripts/convert_mp3_to_wav.py`
`python scripts/convert_human_m4a_to_wav.py`
2. **Normalize Audio**
`python scripts/normalize_audio.py`
3. **Generate Hidden Commands**
`python scripts/generate_hidden_commands.py`
4. **Run Whisper**
`python scripts/run_whisper.py`
5. **Label Results**
Label outputs as: correct, partial, or incorrect.
6. **Analyze Results**
`python scripts/analyze_labeled_results.py`
`python scripts/analyze_labeled_results2.py`
`python scripts/analyze_google_home_results.py`

## Requirements

Install dependencies:
`pip install -r requirements.txt`

### Main Libraries

* pydub
* librosa
* pandas
* matplotlib
* openai-whisper

### FFmpeg Requirement

Verify FFmpeg installation:
`ffmpeg -version`

## Research Focus

This project evaluates:

* perceptual audio masking
* ASR robustness
* smart-assistant activation reliability
* hidden-command behavior under environmental masking

Inspired by:

* CommanderSong
* DolphinAttack
* Light Commands