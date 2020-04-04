from .VGG_SSD import SSD512, MultiBoxLoss


def model_entry(config):
    if config.model['arch'].upper() == 'VGG_SSD':
        print('Loading SSD with VGG16 backbone ......')
        return SSD512(config['n_classes'], device=config.device), MultiBoxLoss
    else:
        raise NotImplementedError
