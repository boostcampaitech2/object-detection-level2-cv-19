import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO
import json
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='WBF ensmeble results')
parser.add_argument('--json_path', type=str, help='json path')
parser.add_argument('--result_name', type=str, default='ensembled_result', help='result json file name')
parser.add_argument('--iou_thr', type=float, default=0.6, help='IOU threshold')
parser.add_argument('--score_thr', type=float, default=0.002, help='confidence threshold')
args = parser.parse_args()

with open('results/weights.txt', 'r') as f:
    lines = f.readlines()

coco = COCO(args.json_path)

preds = []
weights = []

for line in lines:
    
    file_name, weight = line.split()
    file_name = os.path.join('results', file_name if file_name.endswith('.json') else file_name + '.json')

    with open(file_name, 'r') as f:
        annos = json.load(f)
        
    pred = defaultdict(list)
    for anno in annos:
        pred[anno['image_id']].append(anno)
    preds.append(pred)
    
    weights.append(float(weight))
    
image_ids = set()
for pred in preds:
    image_ids.update(pred)
    
ensembled_pred = []

iou_thr = args.iou_thr
score_thr = args.score_thr

for image_id in image_ids:
    
    image_info = coco.loadImgs(0)[0]
    image_width, image_height = image_info['width'], image_info['height']

    boxes_lists = []
    scores_lists = []
    labels_lists = []
    
    for pred in preds:
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for anno in pred[image_id]:
            
            x1, y1, w, h = anno['bbox']
            boxes_list.append([x1/image_width, y1/image_height, (x1+w)/image_width, (y1+h)/image_height])
            scores_list.append(anno['score'])
            labels_list.append(anno['category_id'])

        boxes_lists.append(boxes_list)
        scores_lists.append(scores_list)
        labels_lists.append(labels_list)
    
    if len(boxes_lists):
        boxes, scores, labels = weighted_boxes_fusion(boxes_lists, scores_lists, labels_lists, iou_thr=iou_thr, weights=weights)
        labels = labels.astype(np.int32)
        
        boxes, scores, labels = boxes.tolist(), scores.tolist(), labels.tolist()

        for box, score, label in zip(boxes, scores, labels):

            if score < score_thr:
                continue

            x1, y1, x2, y2 = box

            new_anno = {
                        'image_id': image_id,
                        'bbox': [x1 * image_width, y1*image_height, (x2-x1)*image_width, (y2-y1)*image_height],
                        'score': score,
                        'category_id': label
                        }

            ensembled_pred.append(new_anno)
            
with open(args.result_name + '.json', 'w') as f:
    json.dump(ensembled_pred, f)