import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from map_boxes import mean_average_precision_for_boxes

from dataset import TestDataset
from model import load_net
from transform import get_valid_transform
from func import collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings 파일 위치
    parser.add_argument('--config_dir', type=str, default='./config/valid_settings_base.json')

    args = parser.parse_args()
    print(args)

    with open(args.config_dir, 'r') as outfile:
        settings = (json.load(outfile))

    annotation = settings['annotation']
    data_dir = settings['data']
    val_dataset = TestDataset(annotation, data_dir, get_valid_transform())

    checkpoint_path = settings["check_path"]

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=settings['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = load_net(settings, checkpoint_path)
    model.to(device)

    outputs = []
    for images, image_ids in tqdm(val_data_loader):
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for i, out in enumerate(output):
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1],
                            'image_id': image_ids[i]
            })


    # print(len(outputs))
    predictions = []
    for out in outputs:
        for i in range(len(out['labels'])):
            temp = [
                str(out['image_id']), 
                int(out['labels'][i]) - 1, 
                out['scores'][i], 
                out['boxes'][i][0] * 2, 
                out['boxes'][i][2] * 2, 
                out['boxes'][i][1] * 2, 
                out['boxes'][i][3] * 2
            ]
            predictions.append(temp)
    print(predictions[:3])


    gt = []

    coco = COCO(annotation)
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        annotation_id = coco.getAnnIds(imgIds=image_info['id'])
        annotation_info_list = coco.loadAnns(annotation_id)
            
        file_name = str(image_info['id'])
            
        for annotation in annotation_info_list:
            gt.append([file_name, annotation['category_id'],
                    float(annotation['bbox'][0]),
                    float(annotation['bbox'][0]) + float(annotation['bbox'][2]),
                    float(annotation['bbox'][1]),
                    (float(annotation['bbox'][1]) + float(annotation['bbox'][3]))])
        
    print(gt[:3])


    mean_ap, average_precisions = mean_average_precision_for_boxes(gt, predictions, iou_threshold=0.5)

    print(mean_ap)