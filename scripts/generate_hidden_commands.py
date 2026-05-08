from pathlib import Path
from pydub import AudioSegment
from pydub.effects import low_pass_filter

# ---------------- SETTINGS ----------------

padding_ms = 2000

voice_reduction_light = -18
voice_reduction_strong = -25

# ---------------- PATHS ----------------

input_folder = Path("data/processed/final_normalized")

light_rain_file = Path("data/raw/rain/light_rain.wav")
strong_rain_file = Path("data/raw/rain/strong_rain.wav")

light_output_folder = Path("data/processed/light_rain")
strong_output_folder = Path("data/processed/strong_rain")

light_output_folder.mkdir(parents=True, exist_ok=True)
strong_output_folder.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD RAIN ----------------

light_rain = AudioSegment.from_wav(light_rain_file)
strong_rain = AudioSegment.from_wav(strong_rain_file)

# ---------------- PROCESS ----------------

for wav_file in input_folder.glob("*.wav"):

    print(f"Processing: {wav_file.name}")

    # Load voice
    voice = AudioSegment.from_wav(wav_file)

    # --------------------------------------
    # FILTER SPEECH
    # --------------------------------------

    # Reduce high-frequency clarity
    voice = low_pass_filter(voice, cutoff=2500)

    # Add slight reverb effect
    delayed = voice - 8
    delayed = delayed.fade(from_gain=-120, start=0, duration=50)

    voice = voice.overlay(delayed, position=40)

    # --------------------------------------
    # LIGHT RAIN VERSION
    # --------------------------------------

    total_duration = len(voice) + (padding_ms * 2)

    light_bg = light_rain[:total_duration]

    voice_light = voice + voice_reduction_light

    light_mix = light_bg.overlay(
        voice_light,
        position=padding_ms
    )

    light_output = (
        light_output_folder /
        wav_file.name.replace(".wav", "_light_hidden.wav")
    )

    light_mix.export(light_output, format="wav")

    # --------------------------------------
    # STRONG RAIN VERSION
    # --------------------------------------

    strong_bg = strong_rain[:total_duration]

    voice_strong = voice + voice_reduction_strong

    strong_mix = strong_bg.overlay(
        voice_strong,
        position=padding_ms
    )

    strong_output = (
        strong_output_folder /
        wav_file.name.replace(".wav", "_strong_hidden.wav")
    )

    strong_mix.export(strong_output, format="wav")

    print(f"Finished: {wav_file.name}")

print("Hidden command generation complete.")