# Train 
```
python tools/train.py -f yolox_x.py -b 16 --fp16 -o -c yolox_x.pth
```
## resume train
```
python tools/train.py -f yolox_x.py -b 16 --fp16 -o --resume -c ./YOLOX_outputs/yolox_x/latest_ckpt.pth
```
# inference
``` 
python tools/eval.py -f yolox_x.py -c latest_ckpt.pth -b 8 --tsize 1024
```
## image Demo
``` 
python tools/demo.py image -f yolox_x.py -c latest_ckpt.pth --path "Demo image path" --save_result --conf 0.001
```
# Result
| model | mAP@50 |
|---|---|
|yolox-x| 0.548 |
