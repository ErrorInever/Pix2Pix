import errno
import logging
import torchvision
import wandb
import os
import numpy as np
from matplotlib import pyplot as plt
from config import cfg


logger = logging.getLogger(__name__)


class MetricLogger:
    """Metric class"""
    def __init__(self):
        if cfg.RESUME_ID:
            wandb_id = cfg.WANDB_ID
        else:
            wandb_id = wandb.util.generate_id()

        self.img_subdir_path = f"{os.path.join(cfg.OUT_DIR, cfg.PROJECT_VERSION_NAME)}/examples_images"
        wandb.init(id=wandb_id, project=cfg.PROJECT_NAME, name=cfg.PROJECT_VERSION_NAME, resume=True)
        wandb.config.update({
            'batch_size': cfg.BATCH_SIZE,
            'learning_rate': cfg.LEARNING_RATE,
            'num_epochs': cfg.NUM_EPOCHS
        })

    def log(self, gen_loss, dis_loss, dis_real, dis_fake):
        wandb.log({'dis_loss': dis_loss, 'gen_loss': gen_loss, 'D(x, y)': dis_real, 'D(x, G(x,z))': dis_fake})

    def log_image(self, images, epoch, batch_idx, num_batches, normalize=True):
        # TODO: add save images on hard drive and log
        pass

    @staticmethod
    def _step(epoch, num_batches, batch_idx):
        return epoch * num_batches + batch_idx

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
