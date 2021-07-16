import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Default Pix2Pix convolutional block. ReLU replaced to LeakyReLU"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor([N, C, H, W])``
        """
        return self.conv(x)


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        if features is None:
            self.features = [64, 128, 256, 512]
        else:
            self.features = features
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.body = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.body(x)
        return x


class Block(nn.Module):
    """Default block for generator"""
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        # dropout instead of latent space
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """Generator. UNet arch"""
    def __init__(self, in_channels, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down_1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down_2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down_3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down_4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down_5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down_6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())
        self.up_1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up_2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up_3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up_4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up_5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up_6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up_7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.last_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        down_1 = self.initial_down(x)
        down_2 = self.down_1(down_1)
        down_3 = self.down_2(down_2)
        down_4 = self.down_3(down_3)
        down_5 = self.down_4(down_4)
        down_6 = self.down_5(down_5)
        down_7 = self.down_6(down_6)
        bottleneck = self.bottleneck(down_7)
        up_1 = self.up_1(bottleneck)
        up_2 = self.up_2(torch.cat([up_1, down_7], 1))
        up_3 = self.up_3(torch.cat([up_2, down_6], 1))
        up_4 = self.up_4(torch.cat([up_3, down_5], 1))
        up_5 = self.up_5(torch.cat([up_4, down_4], 1))
        up_6 = self.up_6(torch.cat([up_5, down_3], 1))
        up_7 = self.up_7(torch.cat([up_6, down_2], 1))
        return self.last_up(torch.cat([up_7, down_1], 1))
