#!/usr/bin/env python3
"""
快速开始脚本 - 优化版WGAN-GP
一键训练和评估针对小样本量优化的WGAN-GP模型
"""

import subprocess
import sys
import os
import time


def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {command}")
    print()
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} 完成! 耗时: {end_time - start_time:.1f}秒")
    else:
        print(f"❌ {description} 失败!")
        return False
    
    return True


def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    required_files = [
        'optimized_wgan_gp.py',
        'data_utils.py',
        'fid_score.py',
        'compare_wgan_versions.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    # 检查数据目录
    if not os.path.exists('data'):
        print("❌ 数据目录不存在，请确保HTRU数据集已下载到data/目录")
        return False
    
    print("✅ 所有依赖项检查通过!")
    return True


def main():
    print("🌟 优化版WGAN-GP快速开始脚本")
    print("专门针对小样本量脉冲星数据集优化")
    print()
    
    # 检查依赖项
    if not check_dependencies():
        print("请先解决依赖项问题后再运行此脚本")
        sys.exit(1)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/optimized_wgan_gp', exist_ok=True)
    
    print("📋 训练配置:")
    print("- 模型: 优化版WGAN-GP")
    print("- 训练轮数: 100")
    print("- 批次大小: 16")
    print("- 启用FID评估")
    print("- 启用数据增强")
    print("- 学习率: 0.0001")
    
    # 询问用户是否继续
    response = input("\n是否开始训练? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("训练已取消")
        sys.exit(0)
    
    total_start_time = time.time()
    
    # 步骤1: 训练优化版WGAN-GP
    success = run_command(
        "python optimized_wgan_gp.py --n_epochs 100 --batch_size 16 --enable_fid --data_augmentation --fid_interval 10",
        "训练优化版WGAN-GP模型"
    )
    
    if not success:
        print("训练失败，请检查错误信息")
        sys.exit(1)
    
    # 步骤2: 检查是否有原版WGAN-GP进行比较
    if os.path.exists('results/wgan_gp/generator.pth'):
        print("\n🔍 发现原版WGAN-GP模型，将进行性能比较...")
        
        success = run_command(
            "python compare_wgan_versions.py --num_samples 1000",
            "比较原版和优化版WGAN-GP性能"
        )
        
        if success:
            print("\n📊 性能比较完成!")
            print("- 可视化对比图: results/wgan_gp_versions_comparison.png")
            print("- FID分数对比图: results/wgan_gp_fid_comparison.png")
    else:
        print("\n💡 提示: 如果您想比较原版和优化版的性能，请先训练原版WGAN-GP:")
        print("python train_gans.py --model wgan_gp --n_epochs 30 --enable_fid")
    
    # 步骤3: 生成样本展示
    print("\n🎨 生成样本展示...")
    
    success = run_command(
        "python generate_samples.py --model_type optimized --model_path results/optimized_wgan_gp/generator_ema.pth --num_samples 64",
        "生成优化版WGAN-GP样本"
    )
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n🎉 所有任务完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    print("\n📁 生成的文件:")
    print("- 模型文件: results/optimized_wgan_gp/")
    print("  - generator.pth (普通生成器)")
    print("  - generator_ema.pth (EMA生成器，推荐使用)")
    print("  - critic.pth (评论家)")
    print("- 训练进度图: results/optimized_wgan_gp/training_progress.png")
    print("- 生成样本: results/optimized_wgan_gp/epoch_*.png")
    
    if os.path.exists('results/wgan_gp_versions_comparison.png'):
        print("- 版本对比图: results/wgan_gp_versions_comparison.png")
        print("- FID对比图: results/wgan_gp_fid_comparison.png")
    
    print("\n🔧 优化技术应用:")
    print("✅ 谱归一化 - 提高训练稳定性")
    print("✅ 自注意力机制 - 增强全局一致性")
    print("✅ 指数移动平均 - 稳定生成器权重")
    print("✅ 梯度裁剪 - 防止梯度爆炸")
    print("✅ 学习率调度 - 优化训练动态")
    print("✅ 数据增强 - 增加数据多样性")
    print("✅ Dropout - 防止过拟合")
    
    print("\n💡 使用建议:")
    print("- 对于生成新样本，推荐使用 generator_ema.pth")
    print("- 如果FID分数较高，可以尝试增加训练轮数")
    print("- 可以调整batch_size和学习率进一步优化")
    
    print("\n🚀 下一步:")
    print("1. 查看生成的样本质量")
    print("2. 根据FID分数评估模型性能")
    print("3. 如需要，可以继续训练更多轮次")
    print("4. 使用最佳模型生成更多脉冲星样本")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断了程序")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        sys.exit(1) 