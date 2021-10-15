import json
import numpy as np
from sklearn.model_selection import KFold
import copy
import argparse

parser = argparse.ArgumentParser(description='Create txt list of images')
parser.add_argument('--json_path', type=str, default='', help='json path')
parser.add_argument('--n_splits', type=int, default=5, help='number of splits')
args = parser.parse_args()

with open(args.json_path, 'r') as json_file:
    json_data = json.load(json_file)

kf = KFold(n_splits=args.n_splits, random_state=42, shuffle=True)
X = json_data['images']

for i, (train_index, test_index) in enumerate(kf.split(X), 1):
    
    json_train = copy.deepcopy(json_data)
    json_val = copy.deepcopy(json_data)

    json_train['images'] = [X[j] for j in train_index]
    json_val['images'] = [X[j] for j in test_index]

    imgs = set(d['id'] for d in json_train['images'])
    json_train['annotations']= [d for d in json_train['annotations'] if d['image_id'] in imgs]

    imgs = set(d['id'] for d in json_val['images'])
    json_val['annotations']= [d for d in json_val['annotations'] if d['image_id'] in imgs]
    
    print(len(json_train['images']), len(json_val['images']))
    
    with open(f"train_split{i}.json", "w") as json_file:
        json.dump(json_train, json_file)

    with open(f"val_split{i}.json", "w") as json_file:
        json.dump(json_val, json_file)