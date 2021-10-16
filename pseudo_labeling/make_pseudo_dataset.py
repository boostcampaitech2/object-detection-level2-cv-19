from ensemble_boxes import *
import numpy as np
import argparse
from pycocotools.coco import COCO
import json
import matplotlib.pyplot as plt
from collections import defaultdict


parser = argparse.ArgumentParser(description='WBF ensmeble results')
parser.add_argument('--train_json_path', type=str, help='train json path')
parser.add_argument('--test_json_path', type=str, help='test json path')
parser.add_argument('--result_json_path', type=str, help='prediction json path')
parser.add_argument('--result_name', type=str, default='train_pseudo', help='result json file name')
parser.add_argument('--score_thr', type=float, default=0.5, help='confidence threshold')
args = parser.parse_args()


with open(args.train_json_path, 'r') as f:
    train = json.load(f)

with open(args.test_json_path, 'r') as f:
    test = json.load(f)

with open(args.result_json_path, 'r') as f:
    annos = json.load(f)
    pred = defaultdict(list)
    for anno in annos:
        pred[anno['image_id']].append(anno)
    
score_thres = args.score_thr

image_id = max(img_['id'] for img_ in train['images']) + 1
anno_id = max(img_['id'] for img_ in train['annotations']) + 1

print(f'Before combining: num of images = {len(train["images"])}, num of annos = {len(train["annotations"])}')

for t_img in test['images']:

    new_annos = []

    for anno in pred[t_img['id']]:
        if anno['score'] > score_thres:
            
            x1, y1, w, h = anno['bbox']
            
            new_anno =  {'image_id': image_id,
                         'category_id': anno['category_id'],
                         'area': round(w*h, 2),
                         'bbox': [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                         'iscrowd': 0,
                         'id': anno_id}
            new_annos.append(new_anno)
            
            anno_id += 1

    if len(new_annos) != 0:
        t_img['id'] = image_id
        train['images'].append(t_img)
        train['annotations'].extend(new_annos)
        image_id += 1

print(f'After combining: num of images = {len(train["images"])}, num of annos = {len(train["annotations"])}')

with open(args.result_name + '.json', 'w') as f:
    json.dump(train, f)
