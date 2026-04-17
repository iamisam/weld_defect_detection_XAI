import os
from collections import defaultdict

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\weld_defect_project\datasets\gokul_weld_defects\images"
SPLITS = ["train", "val"]
# ----------------------------------------

def extract_class_name(filename):
    name = filename.split(".")[0]
    if "-" in name:
        return "-".join(name.split("-")[:-1])
    else:
        i = len(name)
        while i > 0 and name[i-1].isdigit():
            i -= 1
        return name[:i]

CLASS_MAP = {
    "air-hole1": "air_hole",
    "air-hole2": "air_hole",
    "slag-inclusion2": "slag_inclusion",
    "bite-edge2": "bite_edge",
    "broken-arc1": "broken_arc",
    "broken-arc2": "broken_arc",
    "crack": "crack",
    "overlap": "overlap",
    "unfused": "lack_of_fusion"
}

def inspect_split(split):
    path = os.path.join(BASE_DIR, split)
    counts = defaultdict(int)

    files = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]

    for f in files:
        base = extract_class_name(f)
        if base in CLASS_MAP:
            counts[CLASS_MAP[base]] += 1

    print(f"\n📂 {split.upper()} DISTRIBUTION")
    total = 0
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")
        total += v
    print("Total:", total)

for s in SPLITS:
    inspect_split(s)