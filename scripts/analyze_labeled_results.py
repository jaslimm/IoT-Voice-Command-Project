from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------

csv_path = Path("data/whisper_results/labeled_results.csv")

df = pd.read_csv(csv_path)

# ---------------- EXTRACT METADATA ----------------

# Voice source

df["voice_source"] = df["file"].apply(
    lambda x: "human" if x.startswith("human") else "tts"
)

# Hidden vs normal

df["hidden"] = df["file"].apply(
    lambda x: "hidden" if "hidden" in x else "normal"
)

# Rain strength

def get_masking(name):
    if "light" in name:
        return "light"
    elif "strong" in name:
        return "strong"
    return "unknown"


df["masking"] = df["file"].apply(get_masking)

# ---------------- SCORE LABELS ----------------

score_map = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0
}


df["score"] = df["label"].map(score_map)

# ---------------- SAVE ENRICHED DATA ----------------

output_csv = Path("data/whisper_results/enriched_results.csv")

df.to_csv(output_csv, index=False)

print(f"Saved enriched dataset to: {output_csv}")

# ---------------- SUMMARY TABLES ----------------

summary = (
    df.groupby(["voice_source", "masking", "hidden"])["score"]
    .mean()
    .reset_index()
)

summary["accuracy_percent"] = summary["score"] * 100

print("\n=== SUMMARY ===")
print(summary)

summary_output = Path("data/whisper_results/summary_results.csv")
summary.to_csv(summary_output, index=False)

print(f"\nSaved summary to: {summary_output}")

# ---------------- GRAPH 1 ----------------
# Hidden vs normal

plt.figure(figsize=(8,5))

for hidden_type in ["normal", "hidden"]:

    subset = summary[summary["hidden"] == hidden_type]

    plt.plot(
        subset["masking"],
        subset["accuracy_percent"],
        marker="o",
        label=hidden_type
    )

plt.title("ASR Accuracy: Hidden vs Normal")
plt.xlabel("Masking Level")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig("graphs/hidden_vs_normal.png")

print("Saved graph: hidden_vs_normal.png")

# ---------------- GRAPH 2 ----------------
# Human vs TTS

plt.figure(figsize=(8,5))

for voice in ["human", "tts"]:

    subset = summary[summary["voice_source"] == voice]

    plt.plot(
        subset["masking"],
        subset["accuracy_percent"],
        marker="o",
        label=voice
    )

plt.title("ASR Accuracy: Human vs TTS")
plt.xlabel("Masking Level")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.savefig("graphs/human_vs_tts.png")

print("Saved graph: human_vs_tts.png")

print("\nAnalysis complete.")
