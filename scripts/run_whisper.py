from pathlib import Path
import whisper
import pandas as pd

# ---------------- LOAD MODEL ----------------

print("Loading Whisper model...")
model = whisper.load_model("base")

# ---------------- INPUT FOLDERS ----------------

folders = [
    Path("data/processed/light_rain"),
    Path("data/processed/strong_rain")
]

# ---------------- RESULTS ----------------

results = []

# ---------------- PROCESS ----------------

for folder in folders:

    print(f"\nChecking folder: {folder}")

    for wav_file in folder.glob("*.wav"):

        print(f"Transcribing: {wav_file.name}")

        result = model.transcribe(str(wav_file))

        transcript = result["text"].strip()

        print(f"Transcript: {transcript}")

        results.append({
            "file": wav_file.name,
            "transcript": transcript
        })

# ---------------- SAVE CSV ----------------

df = pd.DataFrame(results)

output_path = "data/whisper_results/transcripts.csv"

df.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")