from pathlib import Path
from pydub import AudioSegment

input_folder = Path("data/raw/human")
output_folder = Path("data/processed/normalized")

output_folder.mkdir(parents=True, exist_ok=True)

print(list(input_folder.glob("*.m4a")))

for m4a_file in input_folder.glob("*.m4a"):

    print(f"Converting: {m4a_file.name}")

    audio = AudioSegment.from_file(m4a_file, format="m4a")

    # Convert to mono
    audio = audio.set_channels(1)

    # Set sample rate
    audio = audio.set_frame_rate(44100)

    output_file = output_folder / f"{m4a_file.stem}.wav"

    audio.export(output_file, format="wav")

    print(f"Saved: {output_file.name}")

print("Conversion complete.")