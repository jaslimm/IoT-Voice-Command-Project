from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize

input_folder = Path("data/processed/normalized")
output_folder = Path("data/processed/final_normalized")

output_folder.mkdir(parents=True, exist_ok=True)

print(list(input_folder.glob("*.wav")))

for wav_file in input_folder.glob("*.wav"):

    print(f"Normalizing: {wav_file.name}")

    audio = AudioSegment.from_wav(wav_file)

    # Normalize loudness
    normalized_audio = normalize(audio)

    output_file = output_folder / wav_file.name

    normalized_audio.export(output_file, format="wav")

    print(f"Saved: {output_file.name}")

print("Normalization complete.")