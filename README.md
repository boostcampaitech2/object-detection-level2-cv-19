# Boostcamp level2 p-stage
## 재활용품목 분류를 위한 Object Detection - 19조 빨간맛


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

# Installation

# Dataset

# Model

# Train Model

1. [Download Imagenet Pretrained weights](https://github.com/microsoft/Swin-Transformer)

Swin-L 224x224 resolution pretrained weight  
`wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth`

Swin-L 384x384 resolution pretrained weight  
`wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth`

3. Change config file if neccesary
  - data path
  - pretrained weight path
  - etc...

2. Run train script

`python tools/train ./my_config/CONFIG_NAME.py`

# Inference

`python tools/test.py /path/to/config.py /path/to/weight.pth --format-only  --options "jsonfile_prefix=./results"`

## Results

| Model                       | mAP@50 |
|-----------------------------|:------:|
| Swin-L (Baseline_           | 0.614  |
| Swin-L + LabelSmoothing     | 0.614  |
| Swin-L + Mixup              | 0.604  |
| Swin-L + multi-scale + tta  | 0.666  |
