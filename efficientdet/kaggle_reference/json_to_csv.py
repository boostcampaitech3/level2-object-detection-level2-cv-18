from pycocotools.coco import COCO
import pandas as pd
import json

annotation = "../../../detection/dataset/train.json"
with open(annotation, 'r') as outfile:
    annotation_json = (json.load(outfile))

image_id = []
bbox = []
source = []
category_id = []

print(annotation_json['images'][0])

for ann in annotation_json['annotations']:
    image_id.append(ann['image_id'])
    bbox.append(ann['bbox'])
    category_id.append(ann['category_id'])
    source.append(annotation_json['images'][ann['image_id']]['file_name'])

df = pd.DataFrame()
df['image_id'] = image_id
df['bbox'] = bbox
df['source'] = source

df.to_csv('./test.csv', index=None)





