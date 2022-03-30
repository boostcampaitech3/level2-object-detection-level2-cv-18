import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import TestDataset
from model import load_net
from transform import get_valid_transform
from func import collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings 파일 위치
    parser.add_argument('--config_dir', type=str, default='./config/test_settings_base.json')

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


    print(outputs[0])
    # predictions = []
    # for out in outputs:
    #     temp = [out['image_id'], out['labels']]