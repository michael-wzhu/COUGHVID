from functools import partial
from typing import Dict

import timm
import torch
import torchvision.transforms
from timm.models.convnext import LayerNorm2d
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from nnAudio.Spectrogram import STFT

import torchaudio as ta


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


default_config = {'sample_rate': 16000,
                  'window_size': 1024,
                  'n_fft': 1024,
                  'hop_size': 320,
                  'fmin': 50,
                  'fmax': 14000,
                  'mel_bins': 128,
                  'power': 2,
                  'top_db': None}



class TimmClassifier_v1(nn.Module):
    def __init__(self, encoder: str,
                 pretrained=True,
                 classes=2,
                 enable_masking=False,
                 **kwargs
                 ):
        super().__init__()

        print(f"initing CLS features model {kwargs['duration']} duration...")

        mel_config = kwargs['mel_config']
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=mel_config['sample_rate'],
            n_fft=mel_config['window_size'],
            win_length=mel_config['window_size'],
            hop_length=mel_config['hop_size'],
            f_min=mel_config['fmin'],
            f_max=mel_config['fmax'],
            pad=0,
            n_mels=mel_config['mel_bins'],
            power=mel_config['power'],
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=mel_config['top_db'])
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.enable_masking = enable_masking
        if enable_masking:
            self.freq_mask = ta.transforms.FrequencyMasking(24, iid_masks=True)
            self.time_mask = ta.transforms.TimeMasking(64, iid_masks=True)

        self.resize = torchvision.transforms.Resize((224, 224))

        ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        import pathlib
        import timm.models.nfnet as nfnet

        model_name = "eca_nfnet_l0"
        checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()

        nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        print(kwargs['backbone_params'])
        base_model = timm.create_model(
            encoder,
            pretrained=True,
            features_only=False,
            # out_indices=([4]),
            **kwargs['backbone_params']
         )

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        self.head1 = nn.Linear(
            1000,
            classes, bias=True)
        
        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        self.factor = int(wav_crop_len / 5.0)

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):
        # print("x: ", x.shape)
        x = x[:, 0, :] # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, ch, mel, time
            # print("x: ", x.shape)
            x = (x + 80) / 80

        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        print("x: ", x.shape)

        x = self.resize(x)
        # print("x: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x: ", x.shape)
        x = x[:, None, :, :]
        # print("x: ", x.shape)

        encode_outputs = self.encoder(x)
        # print("encode_outputs: ", encode_outputs.shape)

        logit = self.head1(encode_outputs)
        # print("logit: ", logit.shape)
        return {"logit": logit}




