## Train Model

1. Create dataset txt

`python JSON2YOLO/create_dataset_txt.py --json_path /path/to/json`

2. Create label from json

`python JSON2YOLO/general_json2yolo.py`  
`mv JSON2YOLO/build/labels /path/to/dataset/labels`

3. Run train script

`python train.py --img 1024 --batch 10 --epochs 70 --data trash.yaml --name experiment_name --weights yolov5l6.pt --hyp data/hyps/hyp.scratch-p6.yaml --multi-scale`  
`python train.py --img 1024 --batch 6 --epochs 70 --data trash.yaml --name experiment_name --weights yolov5x6.pt --hyp data/hyps/hyp.scratch-p6.yaml --multi-scale`

## Inference

`python val.py --weights /path/to/weights/last.pt --data trash.yaml --img 1024 --iou-thres 0.5 --augment --task test --name experiment_name --save-json`

## Results

| Model                       | mAP@50 |
|----------------------------|:------:|
| yolo5s                     | 0.421  |
| yolo5m                     | 0.467  |
| yolo5l                     | 0.520  |
| yolo5m6                    | 0.546  |
| yolo5l6                    | 0.590  |
| yolo5l6 + ms               | 0.606  |
| yolo5x6 + ms               | 0.621  |
| yolo5l6 + ms + pseudo lableing  | 0.639  |
| yolo5x6 + ms + pseudo lableing  | 0.649  |
