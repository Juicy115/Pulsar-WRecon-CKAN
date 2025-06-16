import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import os

from data_utils import (
    get_pulsar_data, save_generated_images, plot_training_losses,
    count_pulsar_samples, visualize_real_samples, compare_real_vs_generated,
    create_output_dirs, get_all_real_images, plot_training_progress_with_fid
)
from fid_score import FIDEvaluator


class SelfAttention(nn.Module):
    """自注意力机制，提高生成质量"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out


class OptimizedGenerator(nn.Module):
    """优化的生成器，针对小样本量设计"""
    def __init__(self, latent_dim=100, channels=3):
        super(OptimizedGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # 初始全连接层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        # 上采样卷积层
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.2),  # 添加dropout防止过拟合
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            
            # 添加自注意力机制
            SelfAttention(128),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 最终输出层
            nn.ConvTranspose2d(64, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        # 权重初始化
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 512, 4, 4)
        out = self.conv_blocks(out)
        return out


class OptimizedCritic(nn.Module):
    """优化的评论家，使用谱归一化和渐进式增长"""
    def __init__(self, channels=3):
        super(OptimizedCritic, self).__init__()
        
        # 使用谱归一化的卷积层
        self.conv_blocks = nn.Sequential(
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 添加自注意力机制
            SelfAttention(128),
            
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 4x4 -> 2x2
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 最终分类层
        self.fc = spectral_norm(nn.Linear(512 * 2 * 2, 1))
        
        # 权重初始化
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.size(0), -1)
        validity = self.fc(out)
        return validity


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """计算梯度惩罚"""
    batch_size = real_samples.size(0)
    
    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # 计算插值点的评论家输出
    d_interpolates = critic(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class ExponentialMovingAverage:
    """指数移动平均，用于稳定生成器"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_optimized_wgan_gp(dataloader, device, args, fid_evaluator=None):
    """训练优化的WGAN-GP"""
    print("Starting Optimized WGAN-GP training...")
    
    # 初始化模型
    generator = OptimizedGenerator(latent_dim=args.latent_dim).to(device)
    critic = OptimizedCritic().to(device)
    
    # 指数移动平均
    ema_generator = ExponentialMovingAverage(generator, decay=0.999)
    
    # 优化器 - 使用不同的学习率
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr * 0.5, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
    
    # 学习率调度器
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.n_epochs)
    scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=args.n_epochs)
    
    # 训练记录
    losses = {'generator': [], 'discriminator': []}
    fid_scores = []
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(16, args.latent_dim).to(device)
    
    # 训练参数
    lambda_gp = 10.0  # 梯度惩罚系数
    n_critic = 5      # 评论家更新频率
    
    # 数据增强
    def augment_data(images):
        """简单的数据增强"""
        # 随机水平翻转 - 修复维度索引
        if torch.rand(1) > 0.5:
            # 对于形状为[batch, channels, height, width]的图像，width是最后一个维度
            images = torch.flip(images, [-1])  # 使用-1表示最后一个维度
        
        # 随机旋转（小角度）- 简化实现避免维度问题
        if torch.rand(1) > 0.7:
            # 使用90度的倍数旋转避免复杂的角度计算
            k = torch.randint(0, 4, (1,)).item()  # 0, 1, 2, 3对应0°, 90°, 180°, 270°
            if k > 0:
                images = torch.rot90(images, k=k, dims=[-2, -1])  # 在height和width维度上旋转
        
        return images
    
    print(f"Training with {len(dataloader)} batches per epoch")
    
    for epoch in range(args.n_epochs):
        epoch_g_loss = 0
        epoch_c_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.size(0)
            
            # 数据增强
            if args.data_augmentation:
                imgs = augment_data(imgs)
            
            imgs = imgs.to(device)
            
            # ==================
            # 训练评论家
            # ==================
            for _ in range(n_critic):
                optimizer_C.zero_grad()
                
                # 生成图像
                noise = torch.randn(batch_size, args.latent_dim).to(device)
                fake_imgs = generator(noise)
                
                # 评论家损失
                real_validity = critic(imgs)
                fake_validity = critic(fake_imgs.detach())
                
                # 梯度惩罚
                gradient_penalty = compute_gradient_penalty(critic, imgs, fake_imgs, device)
                
                # 总损失
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                
                c_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                
                optimizer_C.step()
                
                epoch_c_loss += c_loss.item()
            
            # ==================
            # 训练生成器
            # ==================
            optimizer_G.zero_grad()
            
            # 生成图像
            noise = torch.randn(batch_size, args.latent_dim).to(device)
            fake_imgs = generator(noise)
            fake_validity = critic(fake_imgs)
            
            g_loss = -torch.mean(fake_validity)
            
            g_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            # 更新EMA
            ema_generator.update()
            
            epoch_g_loss += g_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'C_loss': f'{c_loss.item():.4f}',
                'GP': f'{gradient_penalty.item():.4f}'
            })
        
        # 更新学习率
        scheduler_G.step()
        scheduler_C.step()
        
        # 记录平均损失
        losses['generator'].append(epoch_g_loss / len(dataloader))
        losses['discriminator'].append(epoch_c_loss / (len(dataloader) * n_critic))
        
        # 计算FID分数
        if fid_evaluator and (epoch + 1) % args.fid_interval == 0:
            # 使用EMA生成器进行评估
            ema_generator.apply_shadow()
            
            with torch.no_grad():
                eval_noise = torch.randn(args.fid_samples, args.latent_dim).to(device)
                eval_generated = generator(eval_noise)
                fid_score = fid_evaluator.calculate_fid(eval_generated)
                fid_scores.append(fid_score)
                print(f"Epoch {epoch+1}: FID Score = {fid_score:.2f}")
            
            ema_generator.restore()
        
        # 保存生成的样本
        if (epoch + 1) % args.sample_interval == 0:
            # 使用EMA生成器生成样本
            ema_generator.apply_shadow()
            
            with torch.no_grad():
                fake_imgs = generator(fixed_noise)
                save_generated_images(
                    fake_imgs, 
                    f'results/optimized_wgan_gp/epoch_{epoch+1}.png',
                    nrow=4
                )
            
            ema_generator.restore()
    
    # 保存最终的EMA模型
    ema_generator.apply_shadow()
    torch.save(generator.state_dict(), 'results/optimized_wgan_gp/generator_ema.pth')
    ema_generator.restore()
    
    # 保存普通模型
    torch.save(generator.state_dict(), 'results/optimized_wgan_gp/generator.pth')
    torch.save(critic.state_dict(), 'results/optimized_wgan_gp/critic.pth')
    
    # 绘制损失曲线
    if fid_scores:
        plot_training_progress_with_fid(losses, fid_scores, 'Optimized WGAN-GP', 'results/optimized_wgan_gp/training_progress.png')
    else:
        plot_training_losses(losses, 'results/optimized_wgan_gp/training_losses.png')
    
    return generator, critic, losses, fid_scores


def main():
    parser = argparse.ArgumentParser(description='Train optimized WGAN-GP for small dataset')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (smaller for small dataset)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent space dimension')
    parser.add_argument('--sample_interval', type=int, default=10, help='Sampling interval')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--enable_fid', action='store_true', help='Enable FID evaluation')
    parser.add_argument('--fid_interval', type=int, default=10, help='FID evaluation interval')
    parser.add_argument('--fid_samples', type=int, default=500, help='Number of samples for FID calculation')
    parser.add_argument('--data_augmentation', action='store_true', help='Enable data augmentation')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs('results/optimized_wgan_gp', exist_ok=True)
    
    # 统计脉冲星样本数量
    print("\n=== Dataset Statistics ===")
    count_pulsar_samples(args.data_root)
    
    # 加载数据
    print("\n=== Loading Data ===")
    dataloader = get_pulsar_data(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        train=True
    )
    
    print(f"Pulsar data loaded, number of batches: {len(dataloader)}")
    
    # 可视化真实样本
    print("\n=== Visualizing Real Samples ===")
    visualize_real_samples(dataloader, save_path='results/optimized_wgan_gp/real_samples.png')
    
    # 初始化FID评估器
    fid_evaluator = None
    if args.enable_fid:
        print("\n=== Initializing FID Evaluator ===")
        print("Loading Inception-v3 model for FID calculation...")
        fid_evaluator = FIDEvaluator(device=device, batch_size=min(50, args.batch_size))
        
        # 预计算真实图像的统计信息
        print("Precomputing real image statistics...")
        real_images = get_all_real_images(dataloader)
        fid_evaluator.precompute_real_statistics(real_images)
        print("FID evaluator ready!")
    
    # 训练模型
    print("\n" + "="*60)
    print("Training Optimized WGAN-GP for Small Dataset")
    print("="*60)
    print(f"Optimizations enabled:")
    print(f"- Spectral Normalization: ✓")
    print(f"- Self-Attention: ✓")
    print(f"- Exponential Moving Average: ✓")
    print(f"- Gradient Clipping: ✓")
    print(f"- Learning Rate Scheduling: ✓")
    print(f"- Data Augmentation: {'✓' if args.data_augmentation else '✗'}")
    print("="*60)
    
    start_time = time.time()
    generator, critic, losses, fid_scores = train_optimized_wgan_gp(dataloader, device, args, fid_evaluator)
    end_time = time.time()
    
    print(f"\nOptimized WGAN-GP training completed!")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print("Models and results saved in results/optimized_wgan_gp/ directory")
    
    if fid_scores:
        print(f"Final FID Score: {fid_scores[-1]:.2f}")


if __name__ == '__main__':
    main() 