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
from torchvision.transforms import transforms

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
            # transforms=None,
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
        self.sr = 48000
        self.max_length = 3072

        self.n_classes = n_classes

        # if self.mode == "train":
        #     self.transforms = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                      std=[0.229, 0.224, 0.225]),
        #
        #             ])
        # else:
        #     self.transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225]),
        #
        #     ])
        self.transforms = None

        vc = self.df.label.value_counts()

        self.label2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset,):

        mel_spect = np.load(filename, allow_pickle=True)
        # print("mel_spect: ", mel_spect.shape)

        return mel_spect

    def __getitem__(self, i):
        return self.getitem(i)

    def getitem(self, i):
        row = self.df.iloc[i]
        filename = row['mel_spect_file']
        # print("filename: ", filename)

        ## wav
        offset = 0
        if self.mode == "train":
            wav_len_sec = row["duration_in_seconds"]
            # print(wav_len_sec)
            duration = self.duration
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        elif self.mode == "val":
            offset = 0

        mel_spect = self.load_one(filename, offset=offset)
        if mel_spect is None:
            mel_spect = np.zeros((128, self.max_length), dtype=np.int)
        else:
            if mel_spect.shape[1] < self.max_length:
                mel_spect = np.pad(
                    mel_spect,
                    ((0, 0), (0, self.max_length - mel_spect.shape[1]))
                )
            else:
                mel_spect = mel_spect[:, : self.max_length]

        mel_spect = np.repeat(np.expand_dims(mel_spect, axis=0), 3, axis=0)
        # mel_spect = mel_spect.transpose(1, 2, 0)
        # # print("before transform mel_spect: ", mel_spect.shape)
        # if self.transforms is not None:
        #     mel_spect = self.transforms(mel_spect)
        #     # print("after transform mel_spect: ", mel_spect.shape)

        ## labels
        labels = torch.zeros((self.n_classes,))
        labels[self.label2id[row['label']]] = 1.0

        return {
            "mel_spect": torch.tensor(mel_spect).unsqueeze(0),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    print("run")
    dataset = AudioDataset(
        mode="train",
        folds_csv="./src/TimmSED/folds.csv",
        dataset_dir="./datasets/coughvid_v1/public_dataset",
        fold=0,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0
    )
    for batch in dataloader:
        break
    print(batch['wav'].shape)
