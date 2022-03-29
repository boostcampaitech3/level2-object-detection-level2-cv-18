# Effdet config
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet

import torch
import gc


# Effdet config를 통해 모델 불러오기
def get_net(settings, checkpoint_path=None):
    
    config = get_efficientdet_config(settings['model_name'])
    config.num_classes = settings['num_classes']
    config.image_size = settings['image_size']
    
    if settings['soft_nms'] == 'True':
        config.soft_nms = True
    else:
        config.soft_nms = False
    config.max_det_per_image = settings['max_det_per_image']
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        # print(checkpoint.keys())
        net = DetBenchTrain(net)
        net.load_state_dict(checkpoint)
        return net
    else:
        return DetBenchTrain(net)


# Effdet config를 통해 모델 불러오기 + ckpt load
def load_net(settings, checkpoint_path, device):
    config = get_efficientdet_config(settings['model_name'])
    config.num_classes = settings['num_classes']
    config.image_size = settings['image_size']
    
    if settings['soft_nms'] == 'True':
        config.soft_nms = True
    else:
        config.soft_nms = False
    config.max_det_per_image = settings['max_det_per_image']
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)