from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- LOAD DATA ----------------

csv_path = Path(
    "data/google_home_results/google_home_trials.csv"
)

df = pd.read_csv(csv_path)

# ---------------- SCORE EXECUTION ----------------

execution_score_map = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0
}

wake_score_map = {
    "yes": 1.0,
    "no": 0.0
}

# Normalize text

df["command_executed_correctly"] = (
    df["command_executed_correctly"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df["wake_word_activated"] = (
    df["wake_word_activated"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert to scores

df["execution_score"] = (
    df["command_executed_correctly"]
    .map(execution_score_map)
)


df["wake_score"] = (
    df["wake_word_activated"]
    .map(wake_score_map)
)

# ---------------- GROUP RESULTS ----------------

summary = (
    df.groupby([
        "voice_source",
        "masking_level",
        "hidden"
    ])
    .agg({
        "execution_score": "mean",
        "wake_score": "mean"
    })
    .reset_index()
)

# Convert to percentages

summary["execution_percent"] = (
    summary["execution_score"] * 100
)

summary["wake_percent"] = (
    summary["wake_score"] * 100
)

print("\n=== SUMMARY ===")
print(summary)

# ---------------- CREATE CONDITION LABELS ----------------

summary["condition"] = (
    summary["voice_source"] + "_" +
    summary["masking_level"] + "_" +
    summary["hidden"]
)

# ---------------- GRAPH 1 ----------------
# COMMAND EXECUTION SUCCESS

conditions = summary["condition"]
execution = summary["execution_percent"]
wake = summary["wake_percent"]

x = np.arange(len(conditions))
width = 0.35

plt.figure(figsize=(12,6))

plt.bar(
    x - width/2,
    wake,
    width,
    label="Wake Word Activation"
)

plt.bar(
    x + width/2,
    execution,
    width,
    label="Command Execution"
)

plt.xticks(
    x,
    conditions,
    rotation=45,
    ha="right"
)

plt.ylabel("Success Rate (%)")
plt.xlabel("Condition")
plt.title("Google Home Success Rates by Condition")

plt.legend()

plt.tight_layout()

plt.savefig(
    "graphs/google_home_success_rates.png"
)

print(
    "Saved graph: google_home_success_rates.png"
)

# ---------------- GRAPH 2 ----------------
# HUMAN VS TTS

voice_summary = (
    df.groupby(["voice_source"])
    ["execution_score"]
    .mean()
    .reset_index()
)

voice_summary["execution_percent"] = (
    voice_summary["execution_score"] * 100
)

plt.figure(figsize=(6,5))

plt.bar(
    voice_summary["voice_source"],
    voice_summary["execution_percent"]
)

plt.ylabel("Execution Success Rate (%)")
plt.xlabel("Voice Source")
plt.title("Google Home Success: Human vs TTS")

plt.tight_layout()

plt.savefig(
    "graphs/human_vs_tts_google_home.png"
)

print(
    "Saved graph: human_vs_tts_google_home.png"
)

# ---------------- GRAPH 3 ----------------
# LIGHT VS STRONG

masking_summary = (
    df.groupby(["masking_level"])
    ["execution_score"]
    .mean()
    .reset_index()
)

masking_summary["execution_percent"] = (
    masking_summary["execution_score"] * 100
)

plt.figure(figsize=(6,5))

plt.bar(
    masking_summary["masking_level"],
    masking_summary["execution_percent"]
)

plt.ylabel("Execution Success Rate (%)")
plt.xlabel("Masking Level")
plt.title("Google Home Success: Light vs Strong Masking")

plt.tight_layout()

plt.savefig(
    "graphs/light_vs_strong_google_home.png"
)

print(
    "Saved graph: light_vs_strong_google_home.png"
)

print("\nGoogle Home analysis complete.")