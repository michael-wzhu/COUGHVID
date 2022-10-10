import dataclasses
import logging
import math
import os
import random
import re
from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, List, Union

import librosa
import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from tensorboardX import SummaryWriter
from timm.utils import AverageMeter
from torch.distributions import Beta
from torch.nn import DataParallel, SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import zoo
from src.TimmSED import losses
from src.TimmSED.config_utils import load_config
from src.TimmSED.losses import LossCalculator
from src.TimmSED.training_utils import create_optimizer


@dataclasses.dataclass
class TrainConfiguration:
    config_path: str
    gpu: str = "0"
    distributed: bool = False
    from_zero: bool = True
    zero_score: bool = False
    local_rank: int = 0
    freeze_epochs: int = 0
    test_every: int = 1
    world_size: int = 1
    output_dir: str = "../../weights"
    prefix: str = ""
    resume_checkpoint: str = None
    workers: int = 8
    log_dir: str = "logs"
    fp16: bool = True
    freeze_bn: bool = False
    mixup_prob: float = 0.0


class Evaluator(ABC):
    @abstractmethod
    def init_metrics(self) -> Dict:
        pass

    @abstractmethod
    def validate(self, dataloader: DataLoader, model: torch.nn.Module,
                 local_rank: int = 0, snapshot_name: str = "") -> Dict:
        pass

    @abstractmethod
    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        pass


class LossFunction:

    def __init__(self, loss: LossCalculator, name: str, weight: float = 1, display: bool = False):
        super().__init__()
        self.loss = loss
        self.name = name
        self.weight = weight
        self.display = display


class PytorchInferencer(ABC):
    def __init__(self, train_config: TrainConfiguration,
                 fold: int,
                 ) -> None:
        super().__init__()
        self.fold = fold
        self.train_config = train_config
        self.conf = load_config(train_config.config_path)

        self.model = self._init_model()
        self.model.eval()

    @property
    def train_batch_size(self):
        return self.conf["optimizer"]["train_bs"]

    @property
    def val_batch_size(self):
        return self.conf["optimizer"]["val_bs"]

    @property
    def snapshot_name(self):
        return "{}{}_{}_{}".format(self.train_config.prefix, self.conf["network"],
                                   self.conf["encoder_params"]["encoder"], self.fold)

    def _load_checkpoint(self, model: torch.nn.Module):
        checkpoint_path = self.train_config.resume_checkpoint
        if not checkpoint_path:
            return
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(checkpoint.keys())
            state_dict_name = "state_dict"
            if state_dict_name not in checkpoint:
                state_dict_name = "model"
            if state_dict_name in checkpoint:
                state_dict = checkpoint[state_dict_name]
                state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
                orig_state_dict = model.state_dict()
                mismatched_keys = []
                for k, v in state_dict.items():
                    ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
                    if v.size() != ori_size:
                        print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                        mismatched_keys.append(k)
                for k in mismatched_keys:
                    del state_dict[k]
                model.load_state_dict(state_dict, strict=False)
                # if not self.train_config.from_zero:
                #     self.current_epoch = checkpoint['epoch']
                #     if not self.train_config.zero_score:
                #         self.current_metrics = checkpoint.get('metrics', self.evaluator.init_metrics())
                # print("=> loaded checkpoint '{}' (epoch {})"
                #       .format(checkpoint_path, checkpoint['epoch']))
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
        if self.train_config.from_zero:
            self.current_metrics = self.evaluator.init_metrics()
            self.current_epoch = 0

    def _init_model(self):
        print(self.train_config)

        model = zoo.__dict__[self.conf['network']](**self.conf["encoder_params"])
        # model = model.cuda()
        self._load_checkpoint(model)

        channels_last = self.conf.get("channels_last", False)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def inference_single(self, filename, threshold=0.7):

        self.model.eval()

        # load audio file

        ## wav
        offset = 0
        duration = 12
        sr_gold = 16000
        dsr_gold = duration * sr_gold

        wav, sr = librosa.load(
            filename,
            sr=None,
            offset=offset,
            duration=duration
        )
        if sr != sr_gold:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_gold)
        if wav.shape[0] < dsr_gold:
            wav = np.pad(wav, (0, dsr_gold - wav.shape[0]))
        wav = torch.tensor(wav).unsqueeze(0)
        print("wav: ", wav.shape)

        # wav = wav.unsqueeze(0).cuda()
        wav = wav.unsqueeze(0)

        self.model.eval()
        predictions = self.model(wav, is_test=True)
        print(predictions)

        pred_logits = torch.nn.Softmax(dim=-1)(predictions['logit']).cpu().detach().numpy().tolist()
        print("pred_logits: ", pred_logits)

        if pred_logits[0][1] > threshold:
            pred_label = 1
        else:
            pred_label = 0

        return {
            "predicted_label": pred_label,
            "predicted_logits": pred_logits,
        }






