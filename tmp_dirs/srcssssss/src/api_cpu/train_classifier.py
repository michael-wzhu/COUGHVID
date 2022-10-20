import os
import pandas as pd
import warnings

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120
import os
from typing import Dict

import numpy as np
from sklearn.metrics import classification_report
# from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("./")

from src.api_cpu import metric
# from src.api_cpu import zoo_transforms

from src.api_cpu.config_utils import load_config
from src.api_cpu.losses import tn_score, tp_score
from src.api_cpu.dataset_loader import AudioDataset
from src.api_cpu.trainer import Evaluator, PytorchTrainer, TrainConfiguration


warnings.filterwarnings("ignore")
import argparse


class AudioEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"f1_score": 0, "lb": 0.0, "auc": 0.0}

    def validate(self, dataloader: DataLoader,
                 model: torch.nn.Module,
                 local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        os.makedirs(val_dir, exist_ok=True)

        ## TODO: thresholding?
        val_out = {"gts": [], "preds": []}

        for sample in tqdm(dataloader):
            wav = sample["wav"]
            labels = sample["labels"].numpy()

            model.eval()
            outs = model(wav, is_test=True)
            outs = torch.nn.Softmax(dim=-1)(outs['logit']).cpu().detach().numpy()

            val_out['gts'].extend(labels)
            val_out['preds'].extend(outs)

        val_template = "{conf_name}_val_outs_{local_rank}.npy"
        val_out_path = os.path.join(val_dir, val_template.format(conf_name=conf_name, local_rank=local_rank))
        np.save(val_out_path, val_out)

        best_threshold = -1
        best_f1, best_lb, best_auc = -1, -1, -1
        if self.args.local_rank == 0:
            gts = []
            preds = []
            for rank in range(self.args.world_size):
                val_out_path = os.path.join(val_dir, val_template.format(conf_name=conf_name, local_rank=rank))
                outs = np.load(val_out_path, allow_pickle=True)
                gts.append(np.array(outs[()]['gts']))
                preds.append(np.array(outs[()]['preds']))
            gts = np.concatenate(gts, axis=0)
            preds = np.concatenate(preds, axis=0)
            for threshold in np.arange(0.1, 0.9, 0.05):
                print("threshold: ", threshold)
                tnr = tn_score(torch.from_numpy(preds > threshold).float(), torch.from_numpy(gts))
                tpr = tp_score(torch.from_numpy(preds > threshold).float(), torch.from_numpy(gts))
                print(f"TPR: {tpr.item():0.4f} TNR: {tnr.item():0.4f}")
                lb = float((tpr + tnr) / 2)
                f1s = metric.get_f1(gts, preds, threshold=threshold)
                auc = metric.get_auc(gts, preds, threshold=threshold)
                print(f"f1： {f1s}； auc: {auc}")

                #print(classification_report(gts, preds > threshold, target_names=CLASSES_21))
                if auc > best_auc:
                    best_lb = lb
                    best_f1 = f1s
                    best_auc = auc
                    best_threshold = threshold

        # empty_cache()

        return {
            "f1_score": best_f1,
            "lb": best_lb,
            'auc': best_auc,
            'threshold': best_threshold,
        }

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        for metric in ["f1_score", "lb", "auc"]:
            if current_metrics[metric] > prev_metrics[metric]:
                print("{} improved from {:.6f} to {:.6f}".format(metric, prev_metrics[metric], current_metrics[metric]))
                improved[metric] = current_metrics[metric]
            else:
                print("{} {:.6f} current {:.6f}".format(metric, prev_metrics[metric], current_metrics[metric]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file',
        default="./srcssssss/TimmSED/configs/cls_nf0_v1.json")
    arg('--workers', type=int, default=12, help='number of cpu threads to use PER GPU!')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output_dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='val_')
    arg('--val_dir', type=str, default="validation")
    arg('--data_dir', type=str, default="./datasets/coughvid_v1/public_dataset")
    arg('--folds_csv', type=str, default='srcssssss/TimmSED/data_process/folds_coughvid.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero_score', action='store_true', default=False)
    arg('--from_zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world_size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze_epochs', type=int, default=0)
    arg("--do_train", action='store_true', default=False)
    arg("--do_eval", action='store_true', default=False)
    arg("--freeze_bn", action='store_true', default=False)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)
    print("conf: ", conf)
    train_period = conf["encoder_params"].get("duration") 
    infer_period = conf["encoder_params"].get("val_duration")

    print(f"""
    creating dataset for fold {args.fold}
    transforms                {conf.get("train_transforms")}
    train_period              {train_period}
    infer_period              {infer_period} 
    """)

    # train_transforms = zoo_transforms.__dict__[conf.get("train_transforms")]

    ## set 1 csv
    train_dataset = AudioDataset(
        mode="train",
        folds_csv=args.folds_csv,
        dataset_dir=args.data_dir,
        fold=args.fold,
        multiplier=conf.get("multiplier", 1),
        duration=train_period,
        # transforms=None,
        n_classes=2
    )
    val_dataset = AudioDataset(
        mode="val",
        folds_csv=args.folds_csv,
        dataset_dir=args.data_dir,
        fold=args.fold,
        duration=infer_period,
        n_classes=2
    )
    return train_dataset, val_dataset


def main():
    args = parse_args()
    conf = load_config(args.config)
    print(conf)
    trainer_config = TrainConfiguration(
        config_path=args.config,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn,
        mixup_prob=conf.get("mixup_prob", 0.1)
    )

    data_train, data_val = create_data_datasets(args)

    audio_evaluator = AudioEvaluator(args)
    trainer = PytorchTrainer(
        train_config=trainer_config,
        evaluator=audio_evaluator,
        fold=args.fold,
        train_data=data_train,
        val_data=data_val
    )
    if args.do_train:
        trainer.fit()

    if args.do_eval:
        trainer.validate()



if __name__ == '__main__':
    main()

    # python srcssssss/TimmSED/train_classifier.py --data_dir ./datasets/covid19-cough/raw --folds_csv srcssssss/TimmSED/data_process/folds_covid19_coughs.csv --do_train --do_eval



    # 进行批量的预测
    # python srcssssss/TimmSED/train_classifier.py --data_dir ./datasets/covid19-cough/raw --folds_csv srcssssss/TimmSED/data_process/folds_covid19_coughs.csv --resume weights/val_TimmClassifier_v1_eca_nfnet_l1_0_f1_score --do_eval

