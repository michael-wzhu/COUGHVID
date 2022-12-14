import json
import os
import random

import pandas as pd
from tqdm import tqdm


###############################
# 区别是否是咳嗽音
###############################


if __name__ == "__main__":

    # 有咳嗽的声音
    list_cough_files = []

    # (1) covid19-cough
    df_data_covid19_coughs_1 = pd.read_csv("src/TimmSED/data_process/folds_covid19_coughs.csv")
    list_cough_files.extend(
        list([os.path.join("datasets/covid19-cough/raw", w) for w in df_data_covid19_coughs_1["filename"]])
    )

    # (2) coswara, heavy coughs
    df_data_coswara_1 = pd.read_csv("src/TimmSED/data_process/folds_coswara_shallow.csv")
    list_cough_files.extend(
        list([os.path.join("datasets/coswara", w) for w in df_data_coswara_1["filename"]])
    )
    df_data_coswara_2 = pd.read_csv("src/TimmSED/data_process/folds_coswara_heavy.csv")
    list_cough_files.extend(
        list([os.path.join("datasets/coswara", w) for w in df_data_coswara_2["filename"]])
    )
    print(len(list_cough_files))

    # 没有咳嗽的声音： urbanSound8k
    list_urbanSound8k = []
    for i in range(1, 11):
        folder_ = f"datasets/UrbanSound8K/fold{i}"
        list_files_ = os.listdir(folder_)
        list_files_ = [os.path.join(folder_, w) for w in list_files_]
        # print(list_files_)
        list_urbanSound8k.extend(list_files_)
    print(len(list_urbanSound8k))


    ##############################
    # 构造分类数据集： 是否包含咳嗽音
    ##############################

    df_data = []
    for file in tqdm(list_cough_files):
        file = file.replace("\\", "/")
        label = 1

        fold_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        df_data.append(
            {
                "filename": file,
                "label": label,
                "fold": fold_idx,
            }
        )

    for file in tqdm(list_urbanSound8k):
        file = file.replace("\\", "/")
        label = 0

        fold_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        df_data.append(
            {
                "filename": file,
                "label": label,
                "fold": fold_idx,
            }
        )

    df_data = pd.DataFrame(df_data)
    df_data.to_csv("src/TimmSED/data_process/folds_coughcls.csv", index=False)