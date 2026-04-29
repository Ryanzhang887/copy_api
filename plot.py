import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("copy_eval_outputs/summary.csv")

df = df[df["num_errors"] < df["num_samples"]].copy()

df["adjusted_accuracy"] = (
    df["accuracy"] * df["num_samples"] / (df["num_samples"] - df["num_errors"])
)

# Equal-spacing x positions
target_lengths = sorted(df["target_length"].unique())
x_pos = {length: i for i, length in enumerate(target_lengths)}
df["x_pos"] = df["target_length"].map(x_pos)

plt.figure(figsize=(9, 6))

for model, sub in df.groupby("model"):
    sub = sub.sort_values("target_length")
    plt.plot(
        sub["x_pos"],
        sub["adjusted_accuracy"],
        marker="o",
        linewidth=2,
        label=model,
    )

plt.xlabel("Target length")
plt.ylabel("Adjusted accuracy")
plt.title("Model accuracy vs. target length")

plt.xticks(
    range(len(target_lengths)),
    [str(x) for x in target_lengths],
)

plt.ylim(-0.02, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("accuracy_vs_target_length.png", dpi=300)
plt.show()