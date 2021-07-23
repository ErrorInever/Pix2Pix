import random
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(save_path, gen, dis, opt_gen, opt_dis, lr):
    torch.save({'gen': gen.state_dict(),
                'dis': dis.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_dis': opt_dis.state_dict(),
                'lr': lr},
               save_path)


def create_fixed_batch(idxs, img_names):
    batch = [plt.imread(img_names[i]) for i in idxs]
    batch = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(batch, nrow=1, normalize=True, scale_each=True)
    only_inputs = grid[:, :, 514:]
    only_targets = grid[:, :, :514]
    return grid, only_inputs, only_targets
