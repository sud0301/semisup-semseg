import json

from data.base import *
from data.pcontext_loader import ContextSegmentation
from data.cityscapes_loader import cityscapesLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_context": ContextSegmentation, 
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return './data/city_dataset/'
    if name == 'pascal_context':
        return './data/'
