# Train Model

1. Modify json file
```
python convert_json.py --json_path /path/to/dataset/json_file.json
```
2. Run train script
```
python train.py /path/to/dataset --model tf_efficientdet_d4_ap --dataset trash -b 8 --amp --lr .008 --opt momentum --model-ema --model-ema-decay 0.9966 --epochs 70 --num-classes 10 --pretrained
```
# Inference
```
python validate.py /path/to/dataset --model tf_efficientdet_d4_ap --dataset trash --split test --num-gpu 1 -b 64 --checkpoint /path/to/weights/model_best.pth.tar --num-classes 10 --results result.json
```
# Results

| Model                         | mAP@50 |
|-------------------------------|:------:|
| D0                            | 0.350  |
| D5_ap                         | 0.562  |
| D4_ap + tta                   | 0.602  |
| D5_ap + tta                   | 0.594  |
| D4_ap + tta + pseudo lableing | 0.639  |
| D5_ap + tta + pseudo lableing | 0.642  |
