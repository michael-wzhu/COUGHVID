# coding=utf-8
# Created by Michael Zhu
import argparse
import json

from flask import Flask, request

import sys


sys.path.insert(0, "./")

from src.TimmSED.train_classifier import AudioEvaluator
from src.TimmSED.config_utils import load_config
from src.TimmSED.trainer import Evaluator, TrainConfiguration
from src.TimmSED.inference_api import PytorchInferencer


app = Flask(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file',
        default="./src/TimmSED/configs/cls_nf0_v1.json")
    arg('--workers', type=int, default=12, help='number of cpu threads to use PER GPU!')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output_dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='val_')
    arg('--val_dir', type=str, default="validation")
    arg('--data_dir', type=str, default="./datasets/coughvid_v1/public_dataset")
    arg('--folds_csv', type=str, default='src/TimmSED/data_process/folds_coughvid.csv')
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

    parser.add_argument("--port", default=3001, type=int,
                        help="port number")

    args = parser.parse_args()

    return args



@app.route("/cough_predict", methods=["POST"])
def cough_predict():
    input_data = json.loads(
        request.get_data().decode("utf-8")
    )

    predicted_result = trainer.inference_single(input_data.get("filename"))
    return predicted_result


if __name__ == "__main__":
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
        mixup_prob=conf.get("mixup_prob", 0.5)
    )

    audio_evaluator = AudioEvaluator(args)
    trainer = PytorchInferencer(
        train_config=trainer_config,
        fold=args.fold,
    )

    app.run(host="0.0.0.0", port=args.port, debug=False)

    # python src/TimmSED/flask_service.py --port 9000 --resume ./weights/val_TimmClassifier_v1_eca_nfnet_l1_0_f1_score --config ./src/TimmSED/configs/cls_nf0_v1.json
