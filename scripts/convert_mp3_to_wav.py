from pathlib import Path
from pydub import AudioSegment

input_folder = Path("data/raw/tts")
print(list(input_folder.glob("*.mp3")))
output_folder = Path("data/processed/normalized")

output_folder.mkdir(parents=True, exist_ok=True)

for mp3_file in input_folder.glob("*.mp3"):

    print(f"Converting: {mp3_file.name}")

    audio = AudioSegment.from_mp3(mp3_file)

    # Convert to mono
    audio = audio.set_channels(1)

    # Keep 44.1 kHz
    audio = audio.set_frame_rate(44100)

    output_file = output_folder / f"{mp3_file.stem}.wav"

    audio.export(output_file, format="wav")

    print(f"Saved: {output_file.name}")

print("Conversion complete.")