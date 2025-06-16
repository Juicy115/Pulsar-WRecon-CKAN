import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from gan_models import BasicGenerator, DCGANGenerator, WGANGPGenerator
from data_utils import save_generated_images, denormalize, get_pulsar_data
import os


def load_model(model_type, model_path, latent_dim=100, device='cpu'):
    """
    加载训练好的生成器模型
    
    Args:
        model_type: 模型类型 ('basic_gan', 'dcgan', 'wgan_gp', 'optimized')
        model_path: 模型文件路径
        latent_dim: 潜在空间维度
        device: 计算设备
    
    Returns:
        generator: 加载的生成器模型
    """
    if model_type == 'basic_gan':
        generator = BasicGenerator(latent_dim=latent_dim)
    elif model_type == 'dcgan':
        generator = DCGANGenerator(latent_dim=latent_dim)
    elif model_type == 'wgan_gp':
        generator = WGANGPGenerator(latent_dim=latent_dim)
    elif model_type == 'optimized':
        # 导入优化版生成器
        try:
            from optimized_wgan_gp import OptimizedGenerator
            generator = OptimizedGenerator(latent_dim=latent_dim)
        except ImportError:
            raise ValueError("优化版WGAN-GP模块未找到，请确保optimized_wgan_gp.py文件存在")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型权重
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    return generator


def generate_new_samples(generator, num_samples, latent_dim, device):
    """
    生成新的脉冲星样本
    
    Args:
        generator: 生成器模型
        num_samples: 要生成的样本数量
        latent_dim: 潜在空间维度
        device: 计算设备
    
    Returns:
        generated_images: 生成的图像张量
    """
    with torch.no_grad():
        # 生成随机噪声
        noise = torch.randn(num_samples, latent_dim).to(device)
        
        # 生成图像
        generated_images = generator(noise)
        
    return generated_images


def interpolate_in_latent_space(generator, num_interpolations, latent_dim, device):
    """
    在潜在空间中进行插值，展示生成器的连续性
    
    Args:
        generator: 生成器模型
        num_interpolations: 插值点的数量
        latent_dim: 潜在空间维度
        device: 计算设备
    
    Returns:
        interpolated_images: 插值生成的图像
    """
    with torch.no_grad():
        # 生成两个随机起点
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        
        # 在两点间进行线性插值
        alphas = torch.linspace(0, 1, num_interpolations).to(device)
        
        interpolated_images = []
        for alpha in alphas:
            z = alpha * z2 + (1 - alpha) * z1
            img = generator(z)
            interpolated_images.append(img)
        
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
    return interpolated_images


def compare_models_output(models_info, num_samples, latent_dim, device):
    """
    比较不同模型的生成效果
    
    Args:
        models_info: 模型信息列表，包含模型类型和路径
        num_samples: 每个模型生成的样本数量
        latent_dim: 潜在空间维度
        device: 计算设备
    """
    fig, axes = plt.subplots(len(models_info), num_samples, figsize=(num_samples*2, len(models_info)*2))
    
    # 使用相同的噪声进行公平比较
    fixed_noise = torch.randn(num_samples, latent_dim).to(device)
    
    for i, (model_type, model_path) in enumerate(models_info):
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            continue
            
        try:
            # 加载模型
            generator = load_model(model_type, model_path, latent_dim, device)
            
            # 生成图像
            with torch.no_grad():
                generated_images = generator(fixed_noise)
                generated_images = denormalize(generated_images.clone())
                generated_images = torch.clamp(generated_images, 0, 1)
            
            # 显示图像
            for j in range(num_samples):
                if len(models_info) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                img = generated_images[j].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis('off')
                
                if j == 0:
                    ax.set_ylabel(model_type.upper(), fontsize=12)
        
        except Exception as e:
            print(f"加载模型 {model_type} 时出错: {e}")
    
    plt.suptitle('Comparison of Different GAN Models', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_latent_space_interpolation(generator, model_name, device, latent_dim=100):
    """
    可视化潜在空间插值效果
    """
    # 生成插值图像
    interpolated_images = interpolate_in_latent_space(generator, 10, latent_dim, device)
    
    # 反归一化
    interpolated_images = denormalize(interpolated_images.clone())
    interpolated_images = torch.clamp(interpolated_images, 0, 1)
    
    # 显示插值结果
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    
    for i in range(10):
        img = interpolated_images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Step {i+1}')
    
    plt.suptitle(f'{model_name} - Latent Space Interpolation', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_interpolation.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='使用训练好的GAN模型生成新的脉冲星图像')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['basic_gan', 'dcgan', 'wgan_gp', 'optimized', 'all'],
                        help='选择模型类型')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    parser.add_argument('--num_samples', type=int, default=64, help='生成样本数量')
    parser.add_argument('--latent_dim', type=int, default=100, help='潜在空间维度')
    parser.add_argument('--output_dir', type=str, default='results/generated_samples', 
                        help='输出目录')
    parser.add_argument('--compare_models', action='store_true', 
                        help='比较所有模型的生成效果')
    parser.add_argument('--interpolation', action='store_true',
                        help='显示潜在空间插值效果')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare_models or args.model_type == 'all':
        print("=== 比较所有模型生成效果 ===")
        
        # 模型信息
        models_info = [
            ('basic_gan', 'results/basic_gan/generator.pth'),
            ('dcgan', 'results/dcgan/generator.pth'),
            ('wgan_gp', 'results/wgan_gp/generator.pth'),
            ('optimized', 'results/optimized_wgan_gp/generator_ema.pth'),
        ]
        
        # 比较模型输出
        compare_models_output(models_info, 8, args.latent_dim, device)
        
        # 为每个模型生成插值效果
        if args.interpolation:
            for model_type, model_path in models_info:
                if os.path.exists(model_path):
                    print(f"生成 {model_type} 的插值效果...")
                    generator = load_model(model_type, model_path, args.latent_dim, device)
                    visualize_latent_space_interpolation(generator, model_type, device, args.latent_dim)
    
    else:
        # 单个模型生成
        if not args.model_path:
            # 使用默认路径
            args.model_path = f'results/{args.model_type}/generator.pth'
        
        if not os.path.exists(args.model_path):
            print(f"模型文件不存在: {args.model_path}")
            return
        
        print(f"=== 使用 {args.model_type} 生成新样本 ===")
        
        # 加载模型
        generator = load_model(args.model_type, args.model_path, args.latent_dim, device)
        
        # 生成新样本
        print(f"生成 {args.num_samples} 个新的脉冲星样本...")
        generated_images = generate_new_samples(generator, args.num_samples, args.latent_dim, device)
        
        # 保存生成的图像
        output_path = os.path.join(args.output_dir, f'{args.model_type}_generated_{args.num_samples}.png')
        save_generated_images(generated_images, output_path, nrow=8)
        print(f"生成的样本已保存到: {output_path}")
        
        # 显示部分样本
        generated_images_show = denormalize(generated_images[:16].clone())
        generated_images_show = torch.clamp(generated_images_show, 0, 1)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.ravel()
        
        for i in range(16):
            img = generated_images_show[i].permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        plt.suptitle(f'{args.model_type.upper()} Generated Pulsar Samples', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # 潜在空间插值
        if args.interpolation:
            print("生成潜在空间插值效果...")
            visualize_latent_space_interpolation(generator, args.model_type, device, args.latent_dim)


if __name__ == '__main__':
    main() 