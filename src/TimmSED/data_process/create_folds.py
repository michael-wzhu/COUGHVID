import json
import os
import random

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":


    dict_id2label = json.load(
        open("datasets/coughvid_v1/features/dict_id2label.json", "r", encoding="utf-8")
    )

    folder = "datasets/coughvid_v1/public_dataset"
    list_files = os.listdir(folder)
    import random

    random.shuffle(list_files)

    df_data = []
    for filename in tqdm(list_files):
        print(filename)

        if filename.endswith("json"):
            continue
        if filename.endswith("csv"):
            continue

        idx = filename.split(".")[0]
        if idx not in dict_id2label:
            continue

        label = dict_id2label[idx]

        fold_idx = random.choice([0, 1, 2, 3, 4])
        df_data.append(
            {
                "filename": filename,
                "label": label,
                "fold": fold_idx,
            }
        )

    df_data = pd.DataFrame(df_data)
    df_data.to_csv("src/TimmSED/folds.csv", index=False)