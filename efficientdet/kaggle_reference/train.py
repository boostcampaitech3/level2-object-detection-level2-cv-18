import argparse
import json
import os
from datetime import datetime
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataset import TrainDataset, ValidDataset
from model import get_net
from transform import get_train_transform, get_valid_transform
from calculator import AverageMeter


def collate_fn(batch):
    return tuple(zip(*batch))
    

def save(model, optimizer, scheduler, loss, epoch, path):
    model.eval()
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_summary_loss': loss,
        'epoch': epoch,
    }, path)

    
def log(message, log_path):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # settings 파일 위치
    parser.add_argument('--config_dir', type=str, default='./config/train_settings_base.json')

    args = parser.parse_args()
    print(args)

    with open(args.config_dir, 'r') as outfile:
        settings = (json.load(outfile))

    train_annotation = settings['train_annotation']
    valid_annotation = settings['valid_annotation']
    data_dir = settings['data']

    train_dataset = TrainDataset(train_annotation, data_dir, get_train_transform())
    valid_dataset = ValidDataset(valid_annotation, data_dir, get_valid_transform())

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=settings['batch_size'],
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_data_loader = DataLoader(
        valid_dataset, 
        batch_size=settings['batch_size'],
        num_workers=4,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )


    ########################################################################
    save_dir = settings['save']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_path = f'{save_dir}/log.txt'
    best_summary_loss = 10**5

    ######
    model = get_net(settings)
    device = torch.device('cuda:0')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr_init'])
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    scheduler = SchedulerClass(optimizer, **scheduler_params)
    log(f'Fitter prepared. Device is {device}', log_path)

    num_epochs = settings['epochs']
    print_step = settings['print_step']

    for e in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        timestamp = datetime.utcnow().isoformat()
        log(f'\n{timestamp}\nLR: {lr}', log_path)


        
        ########################### Train
        model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_data_loader):
            if step % print_step == 0:
                now_time = time.time() - t
                print(
                    f'Train Step {step}/{len(train_data_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {int(now_time // 60)}m {int(now_time % 60):02d}s', end='\r'
                )
            
            images = torch.stack(images)
            images = images.to(device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]
            target = {"bbox": boxes, "cls": labels}

            optimizer.zero_grad()
            
            loss, _, _ = model(images, target).values()
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            optimizer.step()


        now_time = time.time() - t
        log(f'[RESULT]: Train. Epoch: {e + 1}, summary_loss: {summary_loss.avg:.5f}, time: {int(now_time // 60)}m {int(now_time % 60)}s', log_path)
        save(
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            loss = summary_loss.avg,
            epoch = e + 1,
            path = f'{save_dir}/last-checkpoint.bin'
        )


        ########################### Valid
        # .eval을 하면 inference mode
        model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_data_loader):
            if step % print_step == 0:
                now_time = time.time() - t
                print(
                    f'Val Step {step}/{len(val_data_loader)}, ' + \
                    f'summary_loss: {summary_loss.avg:.5f}, ' + \
                    f'time: {int(now_time // 60)}m {int(now_time % 60)}s', end='\r'
                )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(device).float()
                boxes = [target['boxes'].to(device).float() for target in targets]
                labels = [target['labels'].to(device).float() for target in targets]
                target = {"bbox": boxes, "cls": labels}

                loss, _, _ = model(images, target).values()
                summary_loss.update(loss.detach().item(), batch_size)


        now_time = time.time() - t
        log(f'[RESULT]: Val. Epoch: {e + 1}, summary_loss: {summary_loss.avg:.5f}, time: {int(now_time // 60)}m {int(now_time % 60)}s', log_path)
        if summary_loss.avg < best_summary_loss:
            best_summary_loss = summary_loss.avg
            model.eval()
            save(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                loss = best_summary_loss,
                epoch = e + 1,
                path = f'{save_dir}/best-checkpoint-{str(e + 1).zfill(3)}epoch.bin'
            )
            for path in sorted(glob(f'{save_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                os.remove(path)

        scheduler.step(metrics=summary_loss.avg)


