import glob
import os
import json
import pandas as pd
from collections import defaultdict

LABEL_DIR = "./datasets/ClipSets"

label_files = glob.glob(f"{LABEL_DIR}/*autotrain*")
label_files.remove(f"{LABEL_DIR}/actions_autotrain.txt")
result = defaultdict(dict)
for label_file in label_files:
    filename = os.path.basename(label_file)
    action_name = filename[: filename.find("_autotrain")]
    with open(label_file, "r") as out:
        for clip_label_pair in out.readlines():
            # Split by double spaces
            [clip_name, label] = clip_label_pair.strip().split("  ", maxsplit=2)
            result[clip_name][action_name] = int(label)
            pass

pd.DataFrame(result).transpose().to_csv("snapshot.csv", header=True, index=True)
