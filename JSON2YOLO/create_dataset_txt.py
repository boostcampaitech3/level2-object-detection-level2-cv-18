import os
import json
import glob
import argparse

parser = argparse.ArgumentParser(description='Create txt list of images')
parser.add_argument('--json_path', type=str, default='', help='json path')
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    annos = json.load(f)
    
with open(args.json_path.replace('.json', '.txt'), 'w') as f:
    
    imgs = []
    for img in annos['images']:
        imgs.append('./images/' + img['file_name'])
    
    f.write('\n'.join(imgs))  