import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from dataset import TrainDataset
from model import get_net
from transform import get_train_transform
from calculator import Averager


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

    annotation = settings['annotation']
    data_dir = settings['data']
    train_dataset = TrainDataset(annotation, data_dir, get_train_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=settings['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    load_epoch = settings['load_epoch']

    if load_epoch > 0:
        check_path = f'{settings["check_path"]}/epoch_{load_epoch}.pth'
    else:
        check_path = None

    model = get_net(settings, check_path)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.1)
    optimizer = torch.optim.AdamW(params)

    num_epochs = settings['epochs']
    clip = settings['clip']

    loss_hist = Averager()
    model.train()
    
    for epoch in range(num_epochs):
        loss_hist.reset()
        
        for images, targets, image_ids in tqdm(train_data_loader):
            
                images = torch.stack(images) # bs, ch, w, h - 16, 3, 512, 512
                images = images.to(device).float()
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels}

                # calculate loss
                loss, cls_loss, box_loss = model(images, target).values()
                loss_value = loss.detach().item()
                
                loss_hist.send(loss_value)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                # grad clip
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()

        print(f"Epoch #{epoch+1+load_epoch} loss: {loss_hist.value}")
        
        save_dir = settings['save']
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_dir + '/' + f'epoch_{epoch+1+load_epoch}.pth')
