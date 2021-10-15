- train : 
python tools/train.py -f yolox_x.py(exp file name) -b 16 --fp16 -o -c yolox_x.pth(pretrained)

- resume train : 
python tools/train.py -f yolox_x.py(exp file name) -b 16 --fp16 -o --resume -c ./YOLOX_outputs/yolox_x/latest_ckpt.pth

- inference : 
python tools/eval.py -f yolox_x.py(exp file name) -c latest_ckpt.pth -b 8 --tsize 1024

- image Demo : 
python tools/demo.py image -f yolox_x.py(exp file name) -c latest_ckpt.pth --path "Demo image path" --save_result --conf 0.001

- Result : 
mAP@50 : 0.548
