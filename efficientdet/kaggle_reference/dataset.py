# base는 baseline 그대로 사용.
# 여기에 kaggle code에만 있는 함수들을 맞추어 적용.
from pycocotools.coco import COCO
import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        
        # coco annotation 불러오기 (by. coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        
        # Cutmix 적용 확률
        if random.random() > 0.5:
            image, boxes, labels, image_id = self.load_image_and_boxes(index)
        else:
            image, boxes, labels, image_id = self.load_cutmix_image_and_boxes(index)
        
        target = {
            'boxes': boxes, 
            'labels': labels, 
            'image_id': torch.tensor([index])
        }

        # transform
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break
            
        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())

    def load_image_and_boxes(self, index):
        image_id = self.coco.getImgIds()[index]

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # boxes (x, y, w, h)
        boxes = np.array([x['bbox'] for x in anns])

        # (x,y,w,h)의 coco format에서 (x_min, y_min, x_max, y_max)로 변경
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # box별 label
        labels = np.array([x['category_id'] + 1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, boxes, labels, image_id

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, len(self.coco.getImgIds()) - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels, _id = self.load_image_and_boxes(index)
            if i == 0:
                image_id = _id
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            np.clip(boxes[:, 0:], 0, 2 * s, out=boxes[:, 0:])
            boxes = boxes.astype(np.int32)
            # 살아남는 box의 index들
            temp = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.where(temp > 0)
            boxes = boxes[idx]
            # 잘라 붙여진 이미지의 너비
            image_area = (x2b - x1b) * (y2b - y1b)
            cutmix_lambda = image_area / (w * h)
            labels = labels[idx] * cutmix_lambda

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        
        return result_image, result_boxes, result_labels, image_id


############################################################################### VALID
class ValidDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        
        # coco annotation 불러오기 (by. coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):

        image, boxes, labels, image_id = self.load_image_and_boxes(index)

        target = {
            'boxes': boxes, 
            'labels': labels, 
            'image_id': torch.tensor([index])
        }

        # transform
        if self.transforms:
            while True:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break
            
        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())

    def load_image_and_boxes(self, index):
        image_id = self.coco.getImgIds()[index]

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # boxes (x, y, w, h)
        boxes = np.array([x['bbox'] for x in anns])

        # (x,y,w,h)의 coco format에서 (x_min, y_min, x_max, y_max)로 변경
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # box별 label
        labels = np.array([x['category_id'] + 1 for x in anns])
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image, boxes, labels, image_id


############################################################################### TEST
class TestDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds()[index]

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # 라벨 등 이미지 외 다른 정보 없기 때문에 train dataset과 달리 이미지만 전처리
        
        # transform
        if self.transforms:
            sample = self.transforms(**{'image': image})

        return sample['image'], image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())