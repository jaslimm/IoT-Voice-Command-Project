from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Masking level
def get_masking(name):
    if "light" in name:
        return "light"
    elif "strong" in name:
        return "strong"
    return "unknown"

df["masking"] = df["file"].apply(get_masking)

# Combined condition label
df["condition"] = (
    df["voice_source"] + "_" +
    df["masking"] + "_" +
    df["hidden"]
)

# ---------------- COUNT LABELS ----------------

label_counts = (
    df.groupby(["condition", "label"])
    .size()
    .unstack(fill_value=0)
)

# Ensure all columns exist
for col in ["correct", "partial", "incorrect"]:
    if col not in label_counts.columns:
        label_counts[col] = 0

# ---------------- GROUPED BAR CHART ----------------

conditions = label_counts.index.tolist()

correct_counts = label_counts["correct"].tolist()
partial_counts = label_counts["partial"].tolist()
incorrect_counts = label_counts["incorrect"].tolist()

x = np.arange(len(conditions))
width = 0.25

plt.figure(figsize=(14,6))

# Side-by-side bars
plt.bar(
    x - width,
    correct_counts,
    width,
    label="Correct"
)

plt.bar(
    x,
    partial_counts,
    width,
    label="Partial"
)

plt.bar(
    x + width,
    incorrect_counts,
    width,
    label="Incorrect"
)

# Labels and formatting
plt.xticks(
    x,
    conditions,
    rotation=45,
    ha="right"
)

plt.ylabel("Number of Samples")
plt.xlabel("Condition")
plt.title("Whisper Recognition Results by Condition")

plt.legend()

plt.tight_layout()

# Save graph
graph_output = "graphs/asr_results_grouped_bar.png"

plt.savefig(graph_output)

print(f"Saved graph: {graph_output}")

plt.show()