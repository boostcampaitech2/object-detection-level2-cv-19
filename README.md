# Boostcamp level2 p-stage - 19조 빨간맛
## Object Detection for classifying recycling item
## (재활용품목 분류를 위한 Object Detection)
![trash with bbox](./trash_with_bbox.png)
# Getting Started
## Dependencies
* albumentations==1.0.3
* pycocotools==2.0.2
* opencv-python==4.5.3.56
* tqdm==4.62.3
* torchnet==0.0.4
* pandas==1.3.3
* map-boxes==1.0.5
* pytorch==1.7.1
* [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
* [UniverseNet](https://github.com/shinya7y/UniverseNet)

## library getting_started
* [Swin Transformer for Object Detection get_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
* [UniverseNet get_started](https://github.com/shinya7y/UniverseNet/blob/master/docs/get_started.md)

# Dataset
## 재활용 쓰레기 데이터셋 / Aistages(Upstage) - CC BY 2.0
This dataset consist of image, classes and bboxes. The number of image is 9754, the number of trainset and is 4883 and 4871, respectively. Also size of image is (1024,1024).

## Class Description
| class | trash |
|---|:-------------:|
| 0 | General trash |
| 1 | Paper         |
| 2 | Paper pack    |
| 3 | Metal         |
| 4 | Glass         |
| 5 | Plastic       |
| 6 | Styrofoam     |
| 7 | Plastic bag   |
| 8 | Battery       |
| 9 | Clothing      |

## Dataset folder path
```
 dataset
 ├── train.json
 ├── test.json
 ├── train
 └── test
```

# Model
| Model                       | mAP@50 |        |
|-----------------------------|:------:|--------|
| Swin-L (Baseline_           | 0.614  | config |
| Swin-L + LabelSmoothing     | 0.614  | config |
| Swin-L + Mixup              | 0.604  | config |
| Swin-L + multi-scale + tta  | 0.666  | config |


# Train Model
## pre-trained model
1. [Download Imagenet Pretrained weights](https://github.com/microsoft/Swin-Transformer)

* Swin-L 224x224 resolution pretrained weight  
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
```

* Swin-L 384x384 resolution pretrained weight  
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
```

2. Change config file if you are necessary
  - data path
  - pretrained weight path
  - etc...

3. Run train script
you can train model usinf config file like below
```
cd library_path
python tools/train ./my_config/CONFIG_NAME.py
```
for example,
```
cd Swin-Transformer-Object-Detection 
python tools/train.py ./configs/level2p/cascade_rcnn_swin-l-p4-w7_fpn_ms_50.py
```
# Inference
```
python tools/test.py /path/to/config.py /path/to/weight.pth --format-only  --options "jsonfile_prefix=./results"
```

