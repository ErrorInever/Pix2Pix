import errno
import logging
import torch
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

        self.img_subdir_path = f"{os.path.join(cfg.OUT_DIR, cfg.PROJECT_VERSION_NAME)}/fixed_images"
        self.img_name = None
        wandb.init(id=wandb_id, project=cfg.PROJECT_NAME, name=cfg.PROJECT_VERSION_NAME, resume=True)
        wandb.config.update({
            'batch_size': cfg.BATCH_SIZE,
            'learning_rate': cfg.LEARNING_RATE,
            'num_epochs': cfg.NUM_EPOCHS
        })

    def log(self, gen_loss, dis_loss, dis_real, dis_fake):
        wandb.log({'dis_loss': dis_loss, 'gen_loss': gen_loss, 'D(x, y)': dis_real, 'D(x, G(x,z))': dis_fake})

    def log_image(self, fixed_image, fixed_fake_y, epoch, batch_idx, normalize=True):
        fixed_fake_y_grid = torchvision.utils.make_grid(fixed_fake_y, nrow=1, normalize=normalize, scale_each=True)
        fixed_image = torch.cat((fixed_image, fixed_fake_y_grid), dim=2)
        wandb.log({'fixed_image': [wandb.Image(np.moveaxis(fixed_image.detach().cpu().numpy(), 0, -1))]})
        img_name = f"epoch{epoch}_step_{batch_idx}.jpg"
        self.save_torch_images(fixed_image, img_name)

    def save_torch_images(self, fixed_image, img_name):
        """
        Display and save image grid
        :param fixed_image: ``numpy ndarray``, fixed image
        :param img_name: ``str``, img name
        """
        out_dir = self.img_subdir_path
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.moveaxis(fixed_image.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        MetricLogger._save_images(fig, out_dir, img_name)
        plt.close()

    @staticmethod
    def _save_images(fig, out_dir, img_name):
        """
        Saves image on drive
        :param fig: plt.figure object
        :param out_dir: path to output dir
        :param img_name: ``str``, grid name for save
        """
        MetricLogger._make_dir(out_dir)
        fig.savefig('{}/{}'.format(out_dir, img_name))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
