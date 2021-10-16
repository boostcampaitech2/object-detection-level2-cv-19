import argparse
import numpy as np
import json
import glob

parser = argparse.ArgumentParser(description='calibrate result scores')
parser.add_argument('--json_path', type=str, help='prediction json path')
parser.add_argument('--result_name', type=str, default='calibrated_result', help='result json file name')
parser.add_argument('--finding_weight', type=float, default=1, help='exponetial weight of finding probability')
parser.add_argument('--class_weight', type=float, default=0.1, help='exponetial weight of class probability')
args = parser.parse_args()

finding_weight = args.finding_weight
class_weight = args.class_weight

results = glob.glob('results/*.pkl')

with open(args.json_path, 'r') as f:
    pred = json.load(f)

clfs = []    
    
for result in results:
    with open(result, 'rb') as f:
        clf = pickle.load(f)
        assert len(clf) == len(pred)
        clfs.append(clf)
    
assert len(clfs) != 0

res_mean = np.mean(clfs, axis=0)

for anno, p in zip(pred, res_mean):
    anno['score'] = anno['score'] * float(p[anno['category_id']])**class_weight * (1-float(p[10]))**finding_weight 
    
with open(args.result_name + '.json', 'w') as f:
    json.dump(pred, f)
