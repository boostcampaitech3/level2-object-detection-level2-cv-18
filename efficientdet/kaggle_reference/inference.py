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
from itertools import product

from dataset import TestDataset
from model import load_net
from transform import get_test_transform
from func import collate_fn
from TTA import *


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
    val_dataset = TestDataset(annotation, data_dir, get_test_transform())

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

    tta_transforms = []
    for tta_combination in product([TTAHorizontalFlip(), None], 
                                [TTAVerticalFlip(), None],
                                [TTARotate90(), TTARotate180(), TTARotate270(), None]):
        tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))

    outputs = []
    for images, image_ids in tqdm(val_data_loader):
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        if settings['TTA']:
            temp_outputs = []
            score_threshold = settings['score_threshold']
            for tta_transform in tta_transforms:
                result = []
                output = model(tta_transform.batch_augment(images.clone()))
                for i, out in enumerate(output):
                    boxes = out.detach().cpu().numpy()[:,:4]
                    scores = out.detach().cpu().numpy()[:,4]
                    labels = out.detach().cpu().numpy()[:,-1]
                    indexes = np.where(scores > score_threshold)[0]
                    boxes = boxes[indexes]
                    boxes = tta_transform.deaugment_boxes(boxes.copy())
                    boxes = (boxes).clip(min=0, max=511).astype(int)
                    result.append({
                        'boxes': boxes,
                        'scores': scores[indexes],
                        'labels': labels[indexes],
                        'image_id': image_ids[i]
                    })
                temp_outputs.append(result)
            for i, image in enumerate(images):
                boxes, scores, labels = run_wbf(temp_outputs, image_index=i)
                boxes = boxes.round().astype(np.int32).clip(min=0, max=511)
                outputs.append({'boxes': boxes, 
                                'scores': scores, 
                                'labels': labels,
                                'image_id': image_ids[i]
                })
        else:
            output = model(images)
            for i, out in enumerate(output):
                outputs.append({'boxes': out.detach().cpu().numpy()[:,:4].clip(min=0, max=511), 
                                'scores': out.detach().cpu().numpy()[:,4], 
                                'labels': out.detach().cpu().numpy()[:,-1],
                                'image_id': image_ids[i]
                })


    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    score_threshold = settings['score_threshold']
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds()[i])[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(int(label) - 1) + ' ' + str(score) + ' ' + str(box[0]* 2) + ' ' + str(
                    box[1] * 2) + ' ' + str(box[2] * 2) + ' ' + str(box[3] * 2) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(f'{settings["save"]}/{settings["file_name"]}', index=None)
    print(submission.head())
