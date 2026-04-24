import os
import pandas as pd
import math

CSV = "D:/weld_defect_project/classification_model/logs/class_distribution_all_splits.csv"
TARGET = 3500

EXCLUDE = {
    "crack",
    "porosity",
    "lack_of_penetration",
    "no_defect"
}

df = pd.read_csv(CSV)

# only train split matters
df = df[df["split"] == "train"]

rows = []

for _, r in df.iterrows():
    cls = r["class"]
    count = r["count"]

    if cls in EXCLUDE:
        rows.append([cls, count, 0, 0])
        continue

    needed = max(0, TARGET - count)

    # augmentations per image
    aug_per_img = math.ceil(needed / count) if count > 0 else 0

    rows.append([cls, count, needed, aug_per_img])

out = pd.DataFrame(rows, columns=[
    "class",
    "current_count",
    "needed_to_3500",
    "augmentations_per_image"
])

print(out.sort_values("current_count"))

out.to_csv(
    "D:/weld_defect_project/classification_model/logs/augmentation_plan.csv",
    index=False
)

print("\nSaved augmentation_plan.csv")