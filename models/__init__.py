from .VGG_SSD import SSD512, MultiBoxLoss
from .VGG_SSD2 import SSD512 as SSD_2
from .VGG_SSD2 import MultiBoxLoss as MBL2
from.VGG_SSD300 import SSD300, MultiBoxLoss300


def model_entry(config):
    if config.model['arch'].upper() == 'VGG_SSD':
        print('Loading SSD with VGG16 backbone ......')
        return SSD512(config['n_classes'], device=config.device), MultiBoxLoss
    elif config.model['arch'].upper() == 'VGG_SSD2':
        print('Loading SSD2 with VGG16 backbone ......')
        return SSD_2(config['n_classes'], device=config.device), MBL2
    elif config.model['arch'].upper() == 'VGG_SSD300':
        print('Loading SSD300 with VGG16 backbone ......')
        return SSD300(config['n_classes'], device=config.device), MultiBoxLoss300
    else:
        raise NotImplementedError
