# Train 
```
bash tools/dist_train.sh  configs/boostcamp/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote.py  1  --work-dir work_0 --seed 0 
```

# Inference 
We make submission file by inference code
```
bash tools/dist_test.sh  configs/boostcamp_trash/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_test_vote.py  pret/latest.pth    1 --eval bbox
```

# Result
| Model        | mAP@50 |
|--------------|:------:|
| UniverseNet  | 0.567  |

