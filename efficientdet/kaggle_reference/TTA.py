# https://www.kaggle.com/code/pohanlin/augmentation-with-efficientdet-d5/notebook
import ensemble_boxes
import torch
import numpy as np


class BaseTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes
    
class TTARotate180(BaseTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 2, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 2, (2, 3))
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,1,2,3]] = self.image_size - boxes[:, [2,3,0,1]]
        return boxes
    
class TTARotate270(BaseTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 3, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 3, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = boxes[:, [1,3]]
        res_boxes[:, [1,3]] = self.image_size - boxes[:, [2,0]]
        return res_boxes
    
class TTACompose(BaseTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


# def make_tta_predictions(images, score_threshold=0.33):
#     tta_transforms = []
#     for tta_combination in product([TTAHorizontalFlip(), None], 
#                                 [TTAVerticalFlip(), None],
#                                 [TTARotate90(), TTARotate180(), TTARotate270(), None]):
#         tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
#     with torch.no_grad():
#         predictions = []
#         for tta_transform in tta_transforms:
#             result = []
#             det = net(tta_transform.batch_augment(images.clone()), torch.tensor([1]*images.shape[0]).float().cuda())

#             for i in range(images.shape[0]):
#                 boxes = det[i].detach().cpu().numpy()[:,:4]    
#                 scores = det[i].detach().cpu().numpy()[:,4]
#                 indexes = np.where(scores > score_threshold)[0]
#                 boxes = boxes[indexes]
#                 boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
#                 boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
#                 boxes = tta_transform.deaugment_boxes(boxes.copy())
#                 result.append({
#                     'boxes': boxes,
#                     'scores': scores[indexes],
#                 })
#             predictions.append(result)
#     return predictions

def run_wbf(predictions, image_index, image_size=512, iou_thr=0.7, skip_box_thr=0.0, weights=None):
    boxes = [(prediction[image_index]['boxes'] / (image_size-1)).tolist() for prediction in predictions]
    # if len(np.where((np.array(boxes) > 1) | (np.array(boxes) < 0))) > 0:
    #     for box in boxes:
    #         print(box)
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [prediction[image_index]['labels'].tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels