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
* [EfficientDet (PyTorch)](https://github.com/rwightman/efficientdet-pytorch)
* [Yolo5](https://github.com/ultralytics/yolov5/)
* [YOloX](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/setup.py)

## Library getting_started
* [Swin Transformer for Object Detection get_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
* [UniverseNet get_started](https://github.com/shinya7y/UniverseNet/blob/master/docs/get_started.md)
* [EfficientDet (PyTorch)](https://github.com/rwightman/efficientdet-pytorch#environment-setup)
* [Yolo5](https://github.com/boostcampaitech2/object-detection-level2-cv-19/blob/jsg_yolo5/yolov5/README_original.md#quick-start-examples)
* [YoloX](https://github.com/Megvii-BaseDetection/YOLOX#quick-start)
)

## Quick mmdetection-based library installation in Ustage Object Detecction server 
* mmdetection 계열이 이미 깔려있는 서버환경에서 빠르게 다른 버전의 mmdetection-based library를 설치 할 수 있다.
* upstage level2 p-stage object detection 서버 사용시 기존 환경을 클론한 환경을 만든다.
```
conda create --name detection_custom --clone detection  # 환경이 설치되어있는 detection 가상환경을 클론
conda activate detection_custom
```
* mmdetection-based library installation
    * mmdetection을 기반으로한 Swin Transformer for Object Detection 설치 예시
```
# Swin Transformer for Object Detection 설치
# https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
cd Swin-Transformer-Object-Detection
pip install -r requirements/build.txt
pip install -v -e .
cd ..
# (optional) Apex설치
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
```


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

# Train Model & Inference
* [Swin-Transformer-Object-Detection](./Swin-Transformer-Object-Detection/README.md)
* [UniverseNet](./UniverseNet/README.md)
* [EfficientDet (PyTorch)](efficientdet-pytorch/README.md)
* [yolov5](yolov5/README.md)
* [yolovX](yolox/README.md)


<!-- clf? notebooks? -->