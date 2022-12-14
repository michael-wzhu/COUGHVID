import math
import os
import ast
import random
import traceback
from builtins import Exception

import librosa

import numpy as np
import pandas as pd
import audiomentations as AA

import torch
from torch.utils.data import Dataset


train_aug = AA.Compose(
    [
        AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        AA.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        AA.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        AA.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ]
)


class AudioDataset(Dataset):
    def __init__(
            self,
            mode: str,
            folds_csv: str,
            dataset_dir: str,
            fold: int = 0,
            n_classes: int = 2,
            transforms=None,
            multiplier: int = 1,
            duration: int = 30,
            # val_duration: int = 5,
    ):
        ## many parts from https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/main/data/ps_ds_2.py
        self.folds_csv = folds_csv
        self.df = pd.read_csv(folds_csv)
        # take sorted labels from full df
        labels = sorted(list(set(self.df.label.values)))
        print('Number of labels ', len(labels))
        if mode =="train":
            self.df = self.df[self.df['fold'] != fold]
        else:
            self.df = self.df[self.df['fold'] == fold]

        self.dataset_dir = dataset_dir
        self.mode = mode

        self.duration = duration
        self.sr = 32000
        self.dsr = self.duration * self.sr

        self.n_classes = n_classes
        self.transforms = transforms

        vc = self.df.label.value_counts()
        dataset_length = len(self.df)
        label_weight = {}
        for row in vc.items():
            label, count = row
            label_weight[label] = math.pow(dataset_length / count, 1/2)

        self.df["label_weight"] = self.df.label.apply(lambda x: label_weight[x])

        self.label2id = {x: idx for idx, x in enumerate(labels)}

        ## TODO: move augmentation assignment outside of dataset
        if self.mode == "train":
            print(f"mode {self.mode} - augmentation is active {train_aug}")
            self.transforms = train_aug
            if multiplier > 1:
                self.df = pd.concat([self.df] * multiplier, ignore_index=True)

    def load_one(self, filename, offset, duration):

        wav, sr = librosa.load(
            filename,
            sr=None,
            offset=offset,
            duration=duration
        )
        try:
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception as e:
            print(e)

        return wav

    def get_weights(self):
        return self.df.label_weight.values

    def __getitem__(self, i):
        tries = 0
        while tries < 20:
            try:
                tries += 1
                return self.getitem(i)
            except:
                traceback.print_stack()
                return self.getitem(random.randint(0, len(self) - 1))
        raise Exception("OOPS, something is wrong!!!")

    def getitem(self, i):
        row = self.df.iloc[i]
        filename = os.path.join(self.dataset_dir, row['filename'])

        ## wav
        if self.mode == "train":
            wav_len_sec = librosa.get_duration(filename=filename)
            duration = self.duration
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        if self.mode == "val":
            offset = 0

        wav = self.load_one(filename, offset=offset, duration=self.duration)
        # print("wav: ", type(wav))
        # if wav is not None:
        #     print("wav: ", wav.shape)

        if wav is None:
            wav = np.zeros((self.dsr, ), dtype=np.int)
        else:
            if wav.shape[0] < self.dsr:
                wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        if self.transforms:
            wav = self.transforms(wav, self.sr)

        ## labels
        labels = torch.zeros((self.n_classes,))
        labels[self.label2id[row['label']]] = 1.0

        ## weight
        weight = 1.0

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
            "weight": weight
        }

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    print("run")
    dataset = AudioDataset(
        mode="train",
        folds_csv="./srcssssss/TimmSED/folds.csv",
        dataset_dir="./datasets/coughvid_v1/public_dataset",
        fold=0,
        transforms=train_aug
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )
    for batch in dataloader:
        break
    print(batch['wav'].shape)
