import argparse
import json
import os

from dataset import TrainDataset, ValidDataset
from model import get_net
from transform import get_train_transform, get_valid_transform


def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings 파일 위치
    parser.add_argument('--config_dir', type=str, default='./config/train_settings_base.json')

    args = parser.parse_args()
    print(args)

    with open(args.config_dir, 'r') as outfile:
        settings = (json.load(outfile))

    test_annotation = settings['test_annotation']
    valid_annotation = settings['valid_annotation']
    data_dir = settings['data']
    train_dataset = TrainDataset(annotation, data_dir, get_train_transform())
    valid_dataset = ValidDataset(annotation, data_dir, get_valid_transform())