import random
from functools import partial
from typing import Dict

import numpy as np

import timm
import torch
from timm.models.convnext import LayerNorm2d
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from nnAudio.Spectrogram import STFT

import torchaudio as ta

from zoo.oned import OneDConvNet


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


# Generalized Mean Pooling (GeM)
#       computes the generalized mean of each channel in a tensor

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(
        x.clamp(min=eps).pow(p),
        (
            x.size(-2),
            x.size(-1)
        )
    ).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
        )


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            print("beware of sigmoid")
            return torch.sigmoid(x)


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

        # ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        # import pathlib
        # import timm.models.nfnet as nfnet
        #
        # model_name = "eca_nfnet_l0"
        # checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        # checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()
        #
        # nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        print(kwargs['backbone_params'])
        base_model = timm.create_model(
            encoder,
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4],
            **kwargs['backbone_params']
         )
        # print(base_model)
        print(base_model.feature_info[-1])
        print(base_model.feature_info[-2])
        print(base_model.feature_info[-3])
        print(base_model.feature_info[-4])
        print(base_model.feature_info[-5])

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        # self.head1 = nn.Linear(
        #     base_model.feature_info[-1]["num_chs"],
        #     classes, bias=True)
        if kwargs.get("cls_head") == "simple":
            self.list_heads = nn.ModuleList(
                [
                    # nn.Linear(base_model.feature_info[-5]["num_chs"], classes, bias=True),
                    # nn.Linear(base_model.feature_info[-4]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-3]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-2]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                ]
            )
        elif kwargs.get("cls_head") == "2layer":

            self.list_heads = nn.ModuleList(
                [
                    # nn.Linear(base_model.feature_info[-5]["num_chs"], classes, bias=True),
                    # nn.Linear(base_model.feature_info[-4]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-3]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-2]["num_chs"], classes, bias=True),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                ]
            )

        elif kwargs.get("cls_head") == "1layer":
            self.list_heads = nn.ModuleList(
                [
                    # nn.Linear(base_model.feature_info[-5]["num_chs"], classes, bias=True),
                    # nn.Linear(base_model.feature_info[-4]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-3]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-2]["num_chs"], classes, bias=True),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                ]
            )

        
        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        # self.factor = int(wav_crop_len / 5.0)
        self.factor = 1

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):

        if  is_test == False:
            x = x[:, 0, :] # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :] # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)   # bs, ch, mel, time
            x = (x + 80) / 80
    
        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # print("x: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x: ", x.shape)
        x = x[:, None, :, :]
        # print("x: ", x.shape)
        
        ## TODO: better loop
        xss = []
        encoded_outputs = self.encoder(x)
        if random.uniform(0, 1) < 0.05:
            for w in encoded_outputs:
                print("feature map: ", w.shape)

        for x in encoded_outputs:
            x = F.dropout(x, p=0.35, training=self.training)
            # print("x: ", x.shape)
            x = self.gem(x)
            # print("x: ", x.shape)
            x = x[:, :, 0, 0]
            # print("x: ", x.shape)
            xss.append(x)

        list_logits = []
        for i, x in enumerate(xss):
            logit_ = self.list_heads[i](x)
            list_logits.append(
                logit_
            )

        last_logit = list_logits[-1]
        return {
            "logit": last_logit,
            "list_logits": list_logits,
        }


class TimmClassifier_v2(nn.Module):
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

        # ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        # import pathlib
        # import timm.models.nfnet as nfnet
        #
        # model_name = "eca_nfnet_l0"
        # checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        # checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()
        #
        # nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        print(kwargs['backbone_params'])
        base_model = timm.create_model(
            encoder,
            pretrained=True,
            features_only=True,
            out_indices=[4],
            **kwargs['backbone_params']
        )
        # print(base_model)
        print(base_model.feature_info[-1])
        print(base_model.feature_info[-2])
        print(base_model.feature_info[-3])
        print(base_model.feature_info[-4])
        print(base_model.feature_info[-5])

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        # self.head1 = nn.Linear(
        #     base_model.feature_info[-1]["num_chs"],
        #     classes, bias=True)


        if kwargs.get("cls_head") == "simple":
            self.list_heads = nn.ModuleList(
                [
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                    nn.Linear(base_model.feature_info[-1]["num_chs"], classes, bias=True),
                ]
            )
        elif kwargs.get("cls_head") == "2layer":

            self.list_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(512, 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.GELU(),
                        nn.Linear(128, classes, bias=True),
                    ),
                ]
            )

        elif kwargs.get("cls_head") == "1layer":

            self.list_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(128, classes, bias=True),
                    ),
                    nn.Sequential(
                        nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                        nn.Dropout(p=0.1, inplace=True),
                        nn.Hardswish(),
                        nn.Linear(128, classes, bias=True),
                    ),
                ]
            )

        else:
            raise ValueError



        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        # self.factor = int(wav_crop_len / 5.0)
        self.factor = 1

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):

        if is_test == False:
            x = x[:, 0, :]  # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :]  # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)  # bs, ch, mel, time
            x = (x + 80) / 80

        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # print("x: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x: ", x.shape)
        x = x[:, None, :, :]
        # print("x: ", x.shape)

        ## TODO: better loop
        xss = []
        encoded_outputs = self.encoder(x)
        if random.uniform(0, 1) < 0.05:
            for w in encoded_outputs:
                print("feature map: ", w.shape)

        # multi-sample dropout
        list_logits = []
        logit_avg = None
        for i, linear_layer in enumerate(self.list_heads):
            feat = encoded_outputs[0]
            feat = F.dropout(feat, p=0.5, training=self.training)

            feat = self.gem(feat)
            feat = feat[:, :, 0, 0]

            logit_ = linear_layer(feat)
            list_logits.append(
                logit_
            )

            if logit_avg is None:
                logit_avg = logit_
            else:
                logit_avg = logit_avg + logit_

        # logit_avg = logit_avg / len(self.list_heads)
        # logit_avg = sum(list_logits)

        # print("logit_avg: ", list_logits)
        # print("list_logits: ", list_logits)
        return {
            "logit": logit_avg,
            "list_logits": list_logits,
        }


class TimmClassifier_v3(nn.Module):
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

        # ## fix https://github.com/rwightman/pytorch-image-models/issues/488#issuecomment-796322390
        # import pathlib
        # import timm.models.nfnet as nfnet
        #
        # model_name = "eca_nfnet_l0"
        # checkpoint_path = "weights/pretrained_eca_nfnet_l0.pth"
        # checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()
        #
        # nfnet.default_cfgs[model_name]["url"] = checkpoint_path_url

        print("pretrained model...")
        print(kwargs['backbone_params'])
        base_model = timm.create_model(
            encoder,
            pretrained=True,
            features_only=True,
            out_indices=[4],
            **kwargs['backbone_params']
        )
        # print(base_model)
        print(base_model.feature_info[-1])
        print(base_model.feature_info[-2])
        print(base_model.feature_info[-3])
        print(base_model.feature_info[-4])
        print(base_model.feature_info[-5])

        self.encoder = base_model

        self.gem = GeM(p=3, eps=1e-6)
        # self.head1 = nn.Linear(
        #     base_model.feature_info[-1]["num_chs"],
        #     classes, bias=True)

        if kwargs.get("cls_head") == "simple":
            self.head1 = nn.Linear(
                base_model.feature_info[-1]["num_chs"],
                classes, bias=True
            )
        elif kwargs.get("cls_head") == "2layer":
            self.head1 = nn.Sequential(
                nn.Linear(base_model.feature_info[-1]["num_chs"], 512, bias=True),
                nn.Dropout(p=0.1, inplace=True),
                nn.Hardswish(),
                nn.Linear(512, 128, bias=True),
                nn.Dropout(p=0.1, inplace=True),
                nn.GELU(),
                nn.Linear(128, classes, bias=True),
            )

        elif kwargs.get("cls_head") == "1layer":

            self.head1 = nn.Sequential(
                nn.Linear(base_model.feature_info[-1]["num_chs"], 128, bias=True),
                nn.Dropout(p=0.1, inplace=True),
                nn.GELU(),
                nn.Linear(128, classes, bias=True),
            )
        else:
            raise ValueError

        ## 30 seconds -> 5 seconds
        wav_crop_len = kwargs["duration"]
        # self.factor = int(wav_crop_len / 5.0)
        self.factor = 1

    ## TODO: optional normalization of mel
    def forward(self, x, is_test=False):

        if is_test == False:
            x = x[:, 0, :]  # bs, ch, time -> bs, time
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
        else:
            ## only 5 seconds infer...
            x = x[:, 0, :]  # bs, ch, time -> bs, time

        with torch.cuda.amp.autocast(enabled=False):
            x = self.wav2img(x)  # bs, ch, mel, time
            x = (x + 80) / 80

        if self.training and self.enable_masking:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # print("x: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x: ", x.shape)
        x = x[:, None, :, :]
        # print("x: ", x.shape)

        ## TODO: better loop
        xss = []
        encoded_outputs = self.encoder(x)
        if random.uniform(0, 1) < 0.05:
            for w in encoded_outputs:
                print("feature map: ", w.shape)

        # multi-sample dropout
        list_logits = []
        logit_avg = None
        for i in range(5):
            feat = encoded_outputs[0]
            feat = F.dropout(feat, p=0.5, training=self.training)

            feat = self.gem(feat)
            feat = feat[:, :, 0, 0]

            logit_ = self.head1(feat)
            list_logits.append(
                logit_
            )

            if logit_avg is None:
                logit_avg = logit_
            else:
                logit_avg = logit_avg + logit_

        # logit_avg = logit_avg / len(self.list_heads)
        # logit_avg = sum(list_logits)

        # print("logit_avg: ", list_logits)
        # print("list_logits: ", list_logits)
        return {
            "logit": logit_avg,
            "list_logits": list_logits,
        }



if __name__ == "__main__":
    net = C1C2(encoder="resnet34", in_chans=1)
    out = net(torch.zeros((1, 1, 32000 * 30)))
    print(out)
