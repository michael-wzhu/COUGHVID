

## 准备工作

```bash

# 安装 ffmpeg
 apt update
 apt install ffmpeg
 ffmpeg -version


```


## 训练

```bash

# 数据集：covid_19_sounds; backbone: mobilenetv3_small_050; 使用TimmClassifier_v2： multi-sample dropout; "cls_head" = "1layer" (1个线性层加激活函数); 
# config: src/TimmSED/configs/cls_mobilenet_covid19sounds_v0.json
python -u src/TimmSED_v1/train_classifier.py --folds_csv src/TimmSED/data_process/folds_covid_19_sounds.csv --config src/TimmSED_v1/configs/cls_mobilenet_covid19sounds_v0.json --output_dir experiments/weights/cls_mobilenet_covid19sounds_v0 --do_train --do_eval

python -u src/TimmSED_v1/train_classifier.py --folds_csv src/TimmSED/data_process/folds_covid_19_sounds.csv --config src/TimmSED_v1/configs/cls_efficientnet_covid19sounds_v0.json --output_dir experiments/weights/cls_efficientnet_covid19sounds_v0 --do_train --do_eval

python -u src/TimmSED_v1/train_classifier.py --folds_csv src/TimmSED/data_process/folds_covid_19_sounds.csv --config src/TimmSED_v1/configs/cls_resnet_covid19sounds_v0.json --output_dir experiments/weights/cls_resnet_covid19sounds_v0_1 --do_train --do_eval




```
