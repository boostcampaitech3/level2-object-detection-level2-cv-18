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
from calculator import Averager


def collate_fn(batch):
    return tuple(zip(*batch))


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

    load_epoch = settings['load_epoch']
    checkpoint_path = f'{settings["check_path"]}/epoch_{load_epoch}.pth'
    
    score_threshold = settings['score_threshold']
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=settings['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = load_net(settings, checkpoint_path, device)
    
    outputs = []
    for images, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)       
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
    
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label) - 1) + ' ' + str(score) + ' ' + str(box[0] * 2) + ' ' + str(
                    box[1] * 2) + ' ' + str(box[2] * 2) + ' ' + str(box[3] * 2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    save_dir = settings['save']
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    submission.to_csv(save_dir + '/' + f'submission_{load_epoch}.csv', index=None)
    print(submission.head())