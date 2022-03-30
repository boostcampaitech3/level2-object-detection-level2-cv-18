from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def get_net(settings):
    config = get_efficientdet_config(settings['model_name'])
    config.num_classes = settings['num_classes']
    config.image_size = settings['image_size']
    config.norm_kwargs=dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=False)

    checkpoint_path = settings['check_path']
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        # print(checkpoint.keys())
        net.load_state_dict(checkpoint)
    
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    return DetBenchTrain(net, config)
    