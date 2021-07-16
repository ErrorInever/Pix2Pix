import logging
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# NAMES
__C.PROJECT_NAME = "Anime sketch colorization"
__C.PROJECT_VERSION_NAME = "Default"
# GLOBAL
__C.IMG_CHANNELS = 3
__C.IMG_SIZE = 256
