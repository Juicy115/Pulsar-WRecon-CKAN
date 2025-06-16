import torch
import argparse
import os
from gan_models import BasicGenerator, DCGANGenerator, WGANGPGenerator
from data_utils import get_pulsar_data, get_all_real_images, plot_fid_comparison
from fid_score import FIDEvaluator


def load_generator(model_type, model_path, latent_dim=100, device='cpu'):
    """加载训练好的生成器模型"""
    if model_type == 'basic_gan':
        generator = BasicGenerator(latent_dim=latent_dim)
    elif model_type == 'dcgan':
        generator = DCGANGenerator(latent_dim=latent_dim)
    elif model_type == 'wgan_gp':
        generator = WGANGPGenerator(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    return generator


def evaluate_single_model(model_type, model_path, fid_evaluator, num_samples, latent_dim, device):
    """评估单个模型的FID分数"""
    print(f"Evaluating {model_type}...")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        generator = load_generator(model_type, model_path, latent_dim, device)
        
        # 生成样本
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim).to(device)
            generated_images = generator(noise)
        
        # 计算FID
        fid_score = fid_evaluator.calculate_fid(generated_images)
        print(f"{model_type} FID Score: {fid_score:.2f}")
        
        return fid_score
        
    except Exception as e:
        print(f"Error evaluating {model_type}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate FID scores for trained GAN models')
    parser.add_argument('--model_type', type=str, default='all', 
                        choices=['basic_gan', 'dcgan', 'wgan_gp', 'all'],
                        help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, help='Path to specific model file')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate for FID')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent space dimension')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载真实数据
    print("Loading real pulsar data...")
    dataloader = get_pulsar_data(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        train=True
    )
    
    # 初始化FID评估器
    print("Initializing FID evaluator...")
    fid_evaluator = FIDEvaluator(device=device, batch_size=min(50, args.batch_size))
    
    # 预计算真实图像统计
    print("Precomputing real image statistics...")
    real_images = get_all_real_images(dataloader)
    fid_evaluator.precompute_real_statistics(real_images)
    print("FID evaluator ready!")
    
    # 评估模型
    if args.model_type == 'all':
        # 评估所有模型
        model_configs = [
            ('basic_gan', 'results/basic_gan/generator.pth'),
            ('dcgan', 'results/dcgan/generator.pth'),
            ('wgan_gp', 'results/wgan_gp/generator.pth'),
        ]
        
        fid_scores = {}
        model_names = []
        
        print("\n=== Evaluating All Models ===")
        for model_type, model_path in model_configs:
            fid_score = evaluate_single_model(
                model_type, model_path, fid_evaluator, 
                args.num_samples, args.latent_dim, device
            )
            if fid_score is not None:
                fid_scores[model_type.upper()] = fid_score
                model_names.append(model_type.upper())
        
        # 绘制比较图
        if fid_scores:
            print(f"\n=== FID Score Summary ===")
            for model, score in fid_scores.items():
                print(f"{model}: {score:.2f}")
            
            # 找出最佳模型
            best_model = min(fid_scores.keys(), key=lambda k: fid_scores[k])
            print(f"\nBest model (lowest FID): {best_model} ({fid_scores[best_model]:.2f})")
            
            # 保存比较图
            fid_values = [fid_scores[name] for name in model_names]
            plot_fid_comparison(fid_values, model_names, 'results/fid_evaluation_comparison.png')
            print("FID comparison chart saved to results/fid_evaluation_comparison.png")
        
    else:
        # 评估单个模型
        if not args.model_path:
            args.model_path = f'results/{args.model_type}/generator.pth'
        
        print(f"\n=== Evaluating {args.model_type.upper()} ===")
        fid_score = evaluate_single_model(
            args.model_type, args.model_path, fid_evaluator,
            args.num_samples, args.latent_dim, device
        )
        
        if fid_score is not None:
            print(f"\nFinal FID Score: {fid_score:.2f}")
        else:
            print("Failed to evaluate model")


if __name__ == '__main__':
    main() 