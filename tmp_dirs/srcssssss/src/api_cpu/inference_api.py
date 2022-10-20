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
from src.api_cpu import losses
from src.api_cpu.config_utils import load_config
from src.api_cpu.losses import LossCalculator
from src.api_cpu.training_utils import create_optimizer


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
                 trainer_config_cough_detect: TrainConfiguration,
                 fold: int,
                 ) -> None:
        super().__init__()
        self.fold = fold
        self.train_config = train_config
        self.trainer_config_cough_detect = trainer_config_cough_detect

        self.conf = load_config(train_config.config_path)
        self.conf_cough_detect = load_config(trainer_config_cough_detect.config_path)

        self.model = self._init_model(
            config=self.conf,
            checkpoint_path=self.train_config.resume_checkpoint)
        self.model.eval()

        self.model_cough_detect = self._init_model(
            config=self.conf_cough_detect,
            checkpoint_path=self.trainer_config_cough_detect.resume_checkpoint
        )
        self.model_cough_detect.eval()

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path=None):
        # checkpoint_path = self.train_config.resume_checkpoint

        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(checkpoint.keys())

        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    def _init_model(self, config=None, checkpoint_path=None):
        # print(self.train_config)

        model = zoo.__dict__[config['network']](**config["encoder_params"])
        # model = model.cuda()
        self._load_checkpoint(model, checkpoint_path=checkpoint_path)

        channels_last = config.get("channels_last", False)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def inference_detect_single(self, filename, threshold_cough_detect=0.5):

        self.model_cough_detect.eval()

        # load audio file

        ## wav
        offset = 0
        duration = 12
        sr_gold = 32000
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
        wav.requires_grad = False

        self.model_cough_detect.eval()
        predictions = self.model_cough_detect(wav, is_test=True)
        print(predictions)

        pred_logits = torch.nn.Softmax(dim=-1)(predictions['logit']).cpu().detach().numpy().tolist()
        print("pred_logits: ", pred_logits)

        if pred_logits[0][1] > threshold_cough_detect:
            pred_label = 1
        else:
            pred_label = 0

        return {
            "has_cough": pred_label,
            "has_cough_predicted_logits": pred_logits[0],
        }


    def inference_single(self, filename, threshold=0.6):

        self.model.eval()

        # load audio file

        ## wav
        offset = 0
        duration = 12
        sr_gold = 32000
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
        wav.requires_grad = False

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
            "cough_is_unhealthy": pred_label,
            "cough_is_unhealthy_predicted_logits": pred_logits[0],
        }

    def inference_detect_and_cls(self, filename, threshold=0.6,
                                 threshold_cough_detect=0.5):

        self.model.eval()
        self.model_cough_detect.eval()

        # load audio file

        ## wav
        offset = 0
        duration = 12
        sr_gold = 32000
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
        wav.requires_grad = False

        # 判断是否有咳嗽音
        self.model_cough_detect.eval()
        predictions_detect = self.model_cough_detect(wav, is_test=True)
        print(predictions_detect)

        pred_logits_detect = torch.nn.Softmax(dim=-1)(predictions_detect['logit']).cpu().detach().numpy().tolist()
        print("pred_logits_detect: ", pred_logits_detect)

        if pred_logits_detect[0][1] > threshold_cough_detect:
            pred_label_detect = 1
        else:
            pred_label_detect = 0


        if pred_label_detect:

            self.model.eval()
            predictions = self.model(wav, is_test=True)
            print(predictions)

            pred_logits = torch.nn.Softmax(dim=-1)(predictions['logit']).cpu().detach().numpy().tolist()
            print("pred_logits: ", pred_logits)

            if pred_logits[0][1] > threshold:
                pred_label = 1
            else:
                pred_label = 0

        else:
            pred_logits = [None]
            pred_label = None


        return {
            "has_cough": pred_label_detect,
            "has_cough_predicted_logits": pred_logits_detect[0],
            "cough_is_unhealthy": pred_label,
            "cough_is_unhealthy_predicted_logits": pred_logits[0],
        }


