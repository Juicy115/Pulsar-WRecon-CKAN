import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 基础GAN模型
# =============================================================================

class BasicGenerator(nn.Module):
    """基础GAN生成器"""
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super(BasicGenerator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class BasicDiscriminator(nn.Module):
    """基础GAN判别器"""
    def __init__(self, img_shape=(3, 32, 32)):
        super(BasicDiscriminator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# =============================================================================
# DCGAN模型
# =============================================================================

class DCGANGenerator(nn.Module):
    """DCGAN生成器"""
    def __init__(self, latent_dim=100, channels=3):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        # 使用转置卷积来上采样
        self.init_size = 4  # 初始尺寸
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),       # 32x32 -> 32x32
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCGANDiscriminator(nn.Module):
    """DCGAN判别器"""
    def __init__(self, channels=3):
        super(DCGANDiscriminator, self).__init__()
        self.channels = channels
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),      # 32x32 -> 16x16
            *discriminator_block(16, 32),                      # 16x16 -> 8x8
            *discriminator_block(32, 64),                      # 8x8 -> 4x4
            *discriminator_block(64, 128),                     # 4x4 -> 2x2
        )

        # 计算卷积后的尺寸
        ds_size = 2
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# =============================================================================
# WGAN-GP模型
# =============================================================================

class WGANGPGenerator(nn.Module):
    """WGAN-GP生成器"""
    def __init__(self, latent_dim=100, channels=3):
        super(WGANGPGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class WGANGPCritic(nn.Module):
    """WGAN-GP判别器(评论家)"""
    def __init__(self, channels=3):
        super(WGANGPCritic, self).__init__()
        self.channels = channels
        
        def critic_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters))  # 使用InstanceNorm而不是BatchNorm
            block.extend([nn.LeakyReLU(0.2, inplace=True)])
            return block

        self.model = nn.Sequential(
            *critic_block(channels, 16, bn=False),
            *critic_block(16, 32),
            *critic_block(32, 64),
            *critic_block(64, 128),
        )

        ds_size = 2
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# =============================================================================
# 梯度惩罚函数 (用于WGAN-GP)
# =============================================================================

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算梯度惩罚"""
    # 随机插值
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# =============================================================================
# 权重初始化函数
# =============================================================================

def weights_init_normal(m):
    """正态分布权重初始化"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 