import os
import pandas as pd

BASE = "D:/weld_defect_project/datasets/unified_classification_standardized_dataset"

splits = ["train", "val", "test"]

rows = []

for split in splits:
    split_path = os.path.join(BASE, split)

    for cls in sorted(os.listdir(split_path)):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue

        count = len([
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ])

        rows.append([split, cls, count])

df = pd.DataFrame(rows, columns=["split", "class", "count"])

# save full table
out_path = "D:/weld_defect_project/classification_model/logs/class_distribution_all_splits.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(df)
print("\nSaved to:", out_path)