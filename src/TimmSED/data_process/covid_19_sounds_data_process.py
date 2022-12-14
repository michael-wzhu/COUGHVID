import hashlib
import json
import os
import random

import librosa

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from librosa.display import specshow
from tqdm import tqdm

if __name__ == "__main__":

    list_files = [
        "datasets/covid_19_sounds/all_metadata/results_raw_20210426_lan_yamnet_android_noloc.csv",
        # "datasets/covid_19_sounds/all_metadata/results_raw_20210426_lan_yamnet_ios_noloc.csv",
        # "datasets/covid_19_sounds/all_metadata/results_raw_20210426_lan_yamnet_web_noloc.csv",
    ]

    # covid_19_sounds:
    df_data = []
    list_symptoms = []
    list_durations = []
    for file_ in list_files:
        df_metadata_ = pd.read_csv(
            file_,
            sep=";"
        )
        # print(df_metadata_ios.head())
        # print(df_metadata_ios.info())

        respiratory_symptoms = [
            "drycough", "wetcough", "fever",
            "sorethroat", "shortbreath", "runnyblockednose",
            "headache", "dizziness", "tightness",
        ]

        for i, row in tqdm(df_metadata_.iterrows()):
            uid = row["Uid"]
            folder_name = row["Folder Name"]
            symptoms = row["Symptoms"]
            cough_filename = row["Cough filename"]
            cough_filename = cough_filename.replace(".m4a", ".wav").replace(".webm", ".wav")
            if cough_filename is None or cough_filename == "None":
                continue

            if pd.isnull(symptoms):
                continue

            list_symptoms.extend(symptoms.split(","))

            if set(symptoms.split(",")).intersection(set(respiratory_symptoms)):
                has_respiratory_symptoms = 1
            else:
                has_respiratory_symptoms = 0

            # print(uid, has_respiratory_symptoms, symptoms, cough_filename, folder_name)

            file_name = f"F:\\covid_19_sounds\\covid19_data_0426\\covid19_data_0426\\{uid}\\{folder_name}\\{cough_filename}"
            # print(file_name)
            if not os.path.isfile(file_name):
                print(file_name)
                continue

            assert os.path.isfile(file_name)

            target_sr = 16000
            duration_in_seconds = 0.0
            wav = None
            try:
                wav, sr = librosa.load(
                    file_name,
                )
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

                duration_in_seconds = librosa.get_duration(y=wav, sr=target_sr)

            except Exception as e:
                print(e)
            if duration_in_seconds == 0.0 or wav is None:
                continue

            print(duration_in_seconds)

            # 将梅尔频谱数据存为np
            mel_spect = librosa.feature.melspectrogram(
                y=wav, sr=target_sr, n_fft=2048, hop_length=1024
            )
            mel_spect = librosa.power_to_db(
                mel_spect, ref=np.max
            )
            # plt.figure()
            # specshow(mel_spect, y_axis="mel", fmax=64000, x_axis="time")
            # plt.title('Mel power spectrogram ')
            # plt.colorbar(format='%+02.0f dB')
            # plt.tight_layout()
            # plt.show()

            mel_spect_file = os.path.join(
                "./datasets/covid_19_sounds/mel_spectrograms",
                hashlib.md5(str(file_name).encode(encoding="utf-8")).hexdigest() + ".npy"
            )
            np.save(
                mel_spect_file,
                mel_spect
            )

            df_data.append(
                {
                    "filename": file_name,
                    "label": has_respiratory_symptoms,
                    "fold": random.choice([0, 1, 2, 3, 4]),
                    "duration_in_seconds": duration_in_seconds,
                    "mel_spect_file": mel_spect_file,
                }
            )

    # print(list_durations)

    df_data = pd.DataFrame(df_data)
    df_data.to_csv("src/TimmSED/data_process/folds_covid_19_sounds__1.csv", index=False)

    # 正负样本比例
    num_samples = len(df_data)
    num_pos_samples = 0
    for i, row in df_data.iterrows():
        if row["label"] == 1:
            num_pos_samples += 1

    print("正样本： ", num_pos_samples)
    print("样本量： ", num_samples)

