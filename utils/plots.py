import pandas as pd
import os
import matplotlib.pyplot as plt


BASE_PLOTS = "D:/weld_defect_project/segmentation_model/plots"
os.makedirs(BASE_PLOTS, exist_ok=True)

df = pd.read_csv("class_distribution_all_splits.csv")

# ---------- BAR PLOTS (log + normal + %) ----------
for split in ["train","val","test"]:
    sub = df[df["split"] == split].sort_values("count", ascending=False)

    # ----- normal -----
    ax = sub.plot(x="class", y="count", kind="bar", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(f"{split} distribution")
    plt.tight_layout()
    plt.savefig(f"{BASE_PLOTS}/class_dist_{split}_normal.png", bbox_inches="tight")
    plt.close()

    # ----- log scale -----
    ax = sub.plot(x="class", y="count", kind="bar", legend=False)
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(f"{split} distribution (log)")
    plt.tight_layout()
    plt.savefig(f"{BASE_PLOTS}/class_dist_{split}_log.png", bbox_inches="tight")
    plt.close()

    # ----- percentage -----
    sub["pct"] = sub["count"] / sub["count"].sum()
    ax = sub.plot(x="class", y="pct", kind="bar", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(f"{split} distribution (%)")
    plt.tight_layout()
    plt.savefig(f"{BASE_PLOTS}/class_dist_{split}_pct.png", bbox_inches="tight")
    plt.close()

# ---------- TABLE ----------
pivot = df.pivot(index="class", columns="split", values="count").fillna(0)
pivot["Total"] = pivot.sum(axis=1)
pivot = pivot[["Total","train","val","test"]]

pivot.to_csv(f"{BASE_PLOTS}/classification_table.csv")

# ---------- TABLE IMAGE ----------
fig, ax = plt.subplots(figsize=(10,6))
ax.axis('off')

table = ax.table(
    cellText=pivot.values,
    colLabels=pivot.columns,
    rowLabels=pivot.index,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Classification Dataset Distribution")
plt.savefig(f"{BASE_PLOTS}/classification_table.png", bbox_inches="tight")
plt.close()

print("Done. Plots + table saved.")

# EMPTY RATION PLOT
df = pd.read_csv("segmentation_empty_stats.csv")

for split in ["train","val","test"]:  # enforce order
    sub = df[df["split"] == split].sort_values("dataset")

    ax = sub.set_index("dataset")[["empty_ratio","non_empty_ratio"]] \
        .plot(kind="bar", stacked=True)

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0,1)  # important (ratios)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.title(f"{split} empty vs non-empty")
    plt.ylabel("ratio")

    plt.tight_layout()
    plt.savefig(f"D:/weld_defect_project/segmentation_model/plots/empty_ratios_{split}.png", bbox_inches="tight")
    plt.close()