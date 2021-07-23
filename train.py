import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from utils import seed_everything, save_checkpoint, create_fixed_batch
from config import cfg
from metric_logger import MetricLogger
from dataset import AnimeSketchColorDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Pix2Pix')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='Path to model weights.pth.tar', default=None, type=str)
    parser.add_argument('--version_name', dest='version_name', help='Version name for wandb', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle, '
                                                              'input api key', default=None, type=str)
    parser.add_argument('--test_epoch', dest='test_epoch', help='Train one epoch for test', action='store_true')
    parser.add_argument('--device', dest='device', help='Use cuda', action='store_true')
    parser.add_argument('--num_epoch', dest='num_epoch', help='Number of epochs', default=None, type=int)
    parser.print_help()
    return parser.parse_args()


def train_one_epoch(gen, dis, gen_opt, dis_opt, g_scaler, d_scaler, criterion, l1, dataloader, metric_logger, epoch,
                    fixed_images, fixed_x):
    """
    :param gen:
    :param dis:
    :param gen_opt:
    :param dis_opt:
    :param g_scaler:
    :param d_scaler:
    :param criterion:
    :param l1:
    :param dataloader:
    :param metric_logger:
    :param epoch:
    :param fixed_images:
    :param fixed_x:
    :return:
    """
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)
        # Train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = dis(x, y)
            d_real_loss = criterion(d_real, torch.ones_like(d_real))
            d_fake = dis(x, y_fake.detach())
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

        dis.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(dis_opt)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = dis(x, y_fake)
            g_fake_loss = criterion(d_fake, torch.ones_like(d_fake))
            l1 = l1_loss(y_fake, y) * cfg.L1_LAMBDA
            g_loss = g_fake_loss + l1

        gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(gen_opt)
        g_scaler.update()

        # metrics
        if batch_idx % 10 == 0:
            loop.set_postfix(
                d_real=torch.sigmoid(d_real).mean().item(),
                d_fake=torch.sigmoid(d_fake).mean().item(),
            )

        if batch_idx % cfg.BATCH_FREQ == 0:
            metric_logger.log(g_loss.item(), d_loss.item(), torch.sigmoid(d_real).mean().item(),
                                  torch.sigmoid(d_fake).mean().item())
        # if batch_idx % cfg.BATCH_IMG_FREQ == 0:
        #     with torch.no_grad():
        #         # logger.info(f"x {x.shape} | fixed_x {fixed_x.shape}")
        #         fixed_fake_y = gen(fixed_x.unsqueeze(0))
        #         metric_logger.log_image(fixed_images, fixed_fake_y, epoch, batch_idx)
        # TODO add metrics eval


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    logger = logging.getLogger('main')
    assert args.data_path, 'data path not specified'
    cfg.DATA_ROOT = args.data_path
    if args.version_name:
        cfg.PROJECT_VERSION_NAME = args.version_name
    else:
        cfg.PROJECT_VERSION_NAME = 'Default_Pix2Pix'
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.wandb_id:
        cfg.WANDB_ID = args.wandb_id
    if args.out_dir:
        cfg.OUT_DIR = args.out_dir
    if args.test_epoch:
        cfg.NUM_EPOCHS = 1
    if args.num_epoch:
        cfg.NUM_EPOCHS = args.num_epoch

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')
    logger.info(f'Using device:{cfg.DEVICE}')

    # data
    train_img_folder = os.path.join(args.data_path, 'train')
    val_img_folder = os.path.join(args.data_path, 'val')

    train_dataset = AnimeSketchColorDataset(train_img_folder)
    val_dataset = AnimeSketchColorDataset(val_img_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True,
                                drop_last=False)

    gen = Generator(in_channels=cfg.IMG_CHANNELS, features=cfg.GEN_FEATURES)
    dis = Discriminator(in_channels=cfg.IMG_CHANNELS)

    # optimizers and amp
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS)
    opt_dis = optim.Adam(dis.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS)

    if args.ckpt:
        try:
            state = torch.load(args.ckpt, map_location=cfg.DEVICE)
            gen.load_state_dict(state['gen'])
            dis.load_state_dict(state['dis'])
            opt_gen.load_state_dict(state['opt_gen'])
            opt_dis.load_state_dict(state['opt_dis'])

            for param_group in opt_gen.param_groups:
                param_group["lr"] = state['lr']
            for param_group in opt_dis.param_groups:
                param_group["lr"] = state['lr']

            logger.info('loaded pretrained weights')
        except Exception:
            logger.error('fail to load model')
            raise ValueError
    else:
        logger.info('loaded default weights')
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    # losses
    criterion = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    metric_logger = MetricLogger()
    idxs = [42, 93, 156, 266, 277]
    train_img_names = [os.path.join(train_img_folder, n) for n in os.listdir(train_img_folder)
                       if n.endswith(('png', 'jpeg', 'jpg'))]
    fixed_images, fixed_x = create_fixed_batch(idxs, train_img_names)

    for epoch in range(cfg.NUM_EPOCHS):
        train_one_epoch(gen, dis, opt_gen, opt_dis, g_scaler, d_scaler, criterion, l1_loss, train_dataloader,
                        metric_logger, epoch, fixed_images, fixed_x)
        if epoch % cfg.SAVE_EPOCH_FREQ == 0 and epoch != 0:
            save_path = os.path.join(cfg.OUT_DIR, f"{cfg.PROJECT_VERSION_NAME}_epoch_{epoch}.pth.tar")
            save_checkpoint(save_path, gen, dis, opt_gen, opt_dis, cfg.LEARNING_RATE)
            logger.info(f"Save model to {save_path}")