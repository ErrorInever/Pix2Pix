import logging
import torch
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NAMES
__C.PROJECT_NAME = "Anime sketch colorization"
__C.PROJECT_VERSION_NAME = "Default"
# GLOBAL
__C.GEN_FEATURES = 64
__C.IMG_CHANNELS = 3
__C.IMG_SIZE = 256
__C.BATCH_SIZE = 4
__C.NUM_EPOCHS = 10
__C.LEARNING_RATE = 1e-4
__C.L1_LAMBDA = 100
# Optimizers
__C.BETAS = (0.5, 0.999)
# NAMES AND PATHS
__C.DATA_ROOT = None
__C.OUTPUT_DIR = '/'
__C.PROJECT_NAME = 'Pix2Pix'
__C.PROJECT_VERSION_NAME = None
# Display
__C.NUM_SAMPLES = 4
__C.BATCH_FREQ = 100
__C.BATCH_IMG_FREQ = 100
# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()

c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)