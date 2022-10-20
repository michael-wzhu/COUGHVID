import json
import os
import random

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    metadata = json.load(
        open("datasets/covid19-cough/metadata.json", "r", encoding="utf-8")
    )

    folder = "datasets/covid19-cough/raw"
    list_files = os.listdir(folder)
    import random

    random.shuffle(list_files)

    df_data = []
    for samp in tqdm(metadata):
        print(samp)

        # label = samp.get("asymptomatic")
        # if label == False:
        #     label = 1
        # else:
        #     label = 0

        label = samp.get("covid19")
        if label:
            label = 1
        else:
            label = 0

        filename = samp["filename"]
        assert filename in list_files

        fold_idx = random.choice([0, 1, 2, 3, 4])
        df_data.append(
            {
                "filename": filename,
                "label": label,
                "fold": fold_idx,
            }
        )

    df_data = pd.DataFrame(df_data)
    df_data.to_csv("src/TimmSED/data_process/folds_covid19_coughs.csv", index=False)