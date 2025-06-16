import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import random

from gan_models import (
    BasicGenerator, BasicDiscriminator,
    DCGANGenerator, DCGANDiscriminator,
    WGANGPGenerator, WGANGPCritic,
    compute_gradient_penalty, weights_init_normal
)
from data_utils import (
    get_pulsar_data, save_generated_images, plot_training_losses,
    count_pulsar_samples, visualize_real_samples, compare_real_vs_generated,
    create_output_dirs, get_all_real_images, plot_fid_comparison, plot_training_progress_with_fid
)
from fid_score import FIDEvaluator


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_basic_gan(dataloader, device, args, fid_evaluator=None):
    """训练基础GAN"""
    print("Starting Basic GAN training...")
    
    # 初始化模型
    generator = BasicGenerator(latent_dim=args.latent_dim).to(device)
    discriminator = BasicDiscriminator().to(device)
    
    # 权重初始化
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # 损失函数和优化器
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 训练记录
    losses = {'generator': [], 'discriminator': []}
    fid_scores = []
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(16, args.latent_dim).to(device)
    
    for epoch in range(args.n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # 真实和虚假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ==================
            # 训练判别器
            # ==================
            optimizer_D.zero_grad()
            
            # 真实图像
            real_pred = discriminator(imgs)
            real_loss = adversarial_loss(real_pred, real_labels)
            
            # 生成图像
            noise = torch.randn(batch_size, args.latent_dim).to(device)
            fake_imgs = generator(noise)
            fake_pred = discriminator(fake_imgs.detach())
            fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # ==================
            # 训练生成器
            # ==================
            optimizer_G.zero_grad()
            
            fake_pred = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_pred, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # 记录平均损失
        losses['generator'].append(epoch_g_loss / len(dataloader))
        losses['discriminator'].append(epoch_d_loss / len(dataloader))
        
        # 计算FID分数
        if fid_evaluator and (epoch + 1) % args.fid_interval == 0:
            with torch.no_grad():
                # 生成更多样本用于FID计算
                eval_noise = torch.randn(args.fid_samples, args.latent_dim).to(device)
                eval_generated = generator(eval_noise)
                fid_score = fid_evaluator.calculate_fid(eval_generated)
                fid_scores.append(fid_score)
                print(f"Epoch {epoch+1}: FID Score = {fid_score:.2f}")
        
        # 保存生成的样本
        if (epoch + 1) % args.sample_interval == 0:
            with torch.no_grad():
                fake_imgs = generator(fixed_noise)
                save_generated_images(
                    fake_imgs, 
                    f'results/basic_gan/epoch_{epoch+1}.png',
                    nrow=4
                )
    
    # 保存模型
    torch.save(generator.state_dict(), 'results/basic_gan/generator.pth')
    torch.save(discriminator.state_dict(), 'results/basic_gan/discriminator.pth')
    
    # 绘制损失曲线
    if fid_scores:
        plot_training_progress_with_fid(losses, fid_scores, 'Basic GAN', 'results/basic_gan/training_progress.png')
    else:
        plot_training_losses(losses, 'results/basic_gan/training_losses.png')
    
    return generator, discriminator, losses, fid_scores


def train_dcgan(dataloader, device, args, fid_evaluator=None):
    """训练DCGAN"""
    print("Starting DCGAN training...")
    
    # 初始化模型
    generator = DCGANGenerator(latent_dim=args.latent_dim).to(device)
    discriminator = DCGANDiscriminator().to(device)
    
    # 权重初始化
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # 损失函数和优化器
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 训练记录
    losses = {'generator': [], 'discriminator': []}
    fid_scores = []
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(16, args.latent_dim).to(device)
    
    for epoch in range(args.n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # 真实和虚假标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ==================
            # 训练判别器
            # ==================
            optimizer_D.zero_grad()
            
            # 真实图像
            real_pred = discriminator(imgs)
            real_loss = adversarial_loss(real_pred, real_labels)
            
            # 生成图像
            noise = torch.randn(batch_size, args.latent_dim).to(device)
            fake_imgs = generator(noise)
            fake_pred = discriminator(fake_imgs.detach())
            fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # ==================
            # 训练生成器
            # ==================
            optimizer_G.zero_grad()
            
            fake_pred = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_pred, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # 记录平均损失
        losses['generator'].append(epoch_g_loss / len(dataloader))
        losses['discriminator'].append(epoch_d_loss / len(dataloader))
        
        # 计算FID分数
        if fid_evaluator and (epoch + 1) % args.fid_interval == 0:
            with torch.no_grad():
                # 生成更多样本用于FID计算
                eval_noise = torch.randn(args.fid_samples, args.latent_dim).to(device)
                eval_generated = generator(eval_noise)
                fid_score = fid_evaluator.calculate_fid(eval_generated)
                fid_scores.append(fid_score)
                print(f"Epoch {epoch+1}: FID Score = {fid_score:.2f}")
        
        # 保存生成的样本
        if (epoch + 1) % args.sample_interval == 0:
            with torch.no_grad():
                fake_imgs = generator(fixed_noise)
                save_generated_images(
                    fake_imgs, 
                    f'results/dcgan/epoch_{epoch+1}.png',
                    nrow=4
                )
    
    # 保存模型
    torch.save(generator.state_dict(), 'results/dcgan/generator.pth')
    torch.save(discriminator.state_dict(), 'results/dcgan/discriminator.pth')
    
    # 绘制损失曲线
    if fid_scores:
        plot_training_progress_with_fid(losses, fid_scores, 'DCGAN', 'results/dcgan/training_progress.png')
    else:
        plot_training_losses(losses, 'results/dcgan/training_losses.png')
    
    return generator, discriminator, losses, fid_scores


def train_wgan_gp(dataloader, device, args, fid_evaluator=None):
    """训练WGAN-GP"""
    print("Starting WGAN-GP training...")
    
    # 初始化模型
    generator = WGANGPGenerator(latent_dim=args.latent_dim).to(device)
    critic = WGANGPCritic().to(device)
    
    # 权重初始化
    generator.apply(weights_init_normal)
    critic.apply(weights_init_normal)
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 训练记录
    losses = {'generator': [], 'discriminator': []}
    fid_scores = []
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(16, args.latent_dim).to(device)
    
    lambda_gp = 10  # 梯度惩罚系数
    n_critic = 5    # 每次生成器更新前的评论家更新次数
    
    for epoch in range(args.n_epochs):
        epoch_g_loss = 0
        epoch_c_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.n_epochs}')
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # ==================
            # 训练评论家
            # ==================
            optimizer_C.zero_grad()
            
            # 生成图像
            noise = torch.randn(batch_size, args.latent_dim).to(device)
            fake_imgs = generator(noise)
            
            # 评论家损失
            real_validity = critic(imgs)
            fake_validity = critic(fake_imgs)
            
            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(critic, imgs, fake_imgs, device)
            
            # 总损失
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            c_loss.backward()
            optimizer_C.step()
            
            epoch_c_loss += c_loss.item()
            
            # ==================
            # 训练生成器
            # ==================
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                # 生成图像
                fake_imgs = generator(noise)
                fake_validity = critic(fake_imgs)
                
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}' if i % n_critic == 0 else 'N/A',
                'C_loss': f'{c_loss.item():.4f}'
            })
        
        # 记录平均损失
        losses['generator'].append(epoch_g_loss / (len(dataloader) // n_critic))
        losses['discriminator'].append(epoch_c_loss / len(dataloader))
        
        # 计算FID分数
        if fid_evaluator and (epoch + 1) % args.fid_interval == 0:
            with torch.no_grad():
                # 生成更多样本用于FID计算
                eval_noise = torch.randn(args.fid_samples, args.latent_dim).to(device)
                eval_generated = generator(eval_noise)
                fid_score = fid_evaluator.calculate_fid(eval_generated)
                fid_scores.append(fid_score)
                print(f"Epoch {epoch+1}: FID Score = {fid_score:.2f}")
        
        # 保存生成的样本
        if (epoch + 1) % args.sample_interval == 0:
            with torch.no_grad():
                fake_imgs = generator(fixed_noise)
                save_generated_images(
                    fake_imgs, 
                    f'results/wgan_gp/epoch_{epoch+1}.png',
                    nrow=4
                )
    
    # 保存模型
    torch.save(generator.state_dict(), 'results/wgan_gp/generator.pth')
    torch.save(critic.state_dict(), 'results/wgan_gp/critic.pth')
    
    # 绘制损失曲线
    if fid_scores:
        plot_training_progress_with_fid(losses, fid_scores, 'WGAN-GP', 'results/wgan_gp/training_progress.png')
    else:
        plot_training_losses(losses, 'results/wgan_gp/training_losses.png')
    
    return generator, critic, losses, fid_scores


def main():
    parser = argparse.ArgumentParser(description='Train GAN models to generate pulsar images')
    parser.add_argument('--model', type=str, default='all', choices=['basic_gan', 'dcgan', 'wgan_gp', 'all'],
                        help='Choose which model to train')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent space dimension')
    parser.add_argument('--sample_interval', type=int, default=10, help='Sampling interval')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--enable_fid', action='store_true', help='Enable FID evaluation')
    parser.add_argument('--fid_interval', type=int, default=5, help='FID evaluation interval')
    parser.add_argument('--fid_samples', type=int, default=1000, help='Number of samples for FID calculation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    create_output_dirs()
    
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
    visualize_real_samples(dataloader, save_path='results/real_samples.png')
    
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
    
    # 存储所有模型的FID分数用于比较
    all_fid_scores = {}
    model_names = []
    
    # 训练模型
    if args.model == 'basic_gan' or args.model == 'all':
        print("\n" + "="*50)
        print("Training Basic GAN")
        print("="*50)
        start_time = time.time()
        gen_basic, disc_basic, losses_basic, fid_scores_basic = train_basic_gan(dataloader, device, args, fid_evaluator)
        end_time = time.time()
        print(f"Basic GAN training completed, time: {end_time - start_time:.2f} seconds")
        if fid_scores_basic:
            all_fid_scores['Basic GAN'] = fid_scores_basic[-1]  # 最后的FID分数
            model_names.append('Basic GAN')
    
    if args.model == 'dcgan' or args.model == 'all':
        print("\n" + "="*50)
        print("Training DCGAN")
        print("="*50)
        start_time = time.time()
        gen_dcgan, disc_dcgan, losses_dcgan, fid_scores_dcgan = train_dcgan(dataloader, device, args, fid_evaluator)
        end_time = time.time()
        print(f"DCGAN training completed, time: {end_time - start_time:.2f} seconds")
        if fid_scores_dcgan:
            all_fid_scores['DCGAN'] = fid_scores_dcgan[-1]
            model_names.append('DCGAN')
    
    if args.model == 'wgan_gp' or args.model == 'all':
        print("\n" + "="*50)
        print("Training WGAN-GP")
        print("="*50)
        start_time = time.time()
        gen_wgan, critic_wgan, losses_wgan, fid_scores_wgan = train_wgan_gp(dataloader, device, args, fid_evaluator)
        end_time = time.time()
        print(f"WGAN-GP training completed, time: {end_time - start_time:.2f} seconds")
        if fid_scores_wgan:
            all_fid_scores['WGAN-GP'] = fid_scores_wgan[-1]
            model_names.append('WGAN-GP')
    
    # 绘制FID比较图
    if all_fid_scores:
        print(f"\n=== FID Score Comparison ===")
        for model, score in all_fid_scores.items():
            print(f"{model}: {score:.2f}")
        
        fid_values = [all_fid_scores[name] for name in model_names]
        plot_fid_comparison(fid_values, model_names, 'results/fid_comparison.png')
    
    print("\n=== All models training completed ===")
    print("Generated images and models saved in results/ directory")
    if args.enable_fid:
        print("FID scores calculated and comparison chart saved")


if __name__ == '__main__':
    main() 