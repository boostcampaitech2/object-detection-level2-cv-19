## Train Model

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

## Inference

`python tools/test.py /path/to/config.py /path/to/weight.pth --format-only  --options "jsonfile_prefix=./results"`

## Results

| Model                       | mAP@50 |
|-----------------------------|:------:|
| Swin-L (Baseline_           | 0.614  |
| Swin-L + LabelSmoothing     | 0.614  |
| Swin-L + Mixup              | 0.604  |
| Swin-L + multi-scale + tta  | 0.666  |
