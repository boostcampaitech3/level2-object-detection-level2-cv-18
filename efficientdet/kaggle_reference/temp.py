from pycocotools.coco import COCO
import pandas as pd
import json

annotation = "../../../detection/dataset/train.json"
with open(annotation, 'r') as outfile:
        annotation_json = (json.load(outfile))

print(annotation_json['annotations'][0])


