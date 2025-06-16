# compare_models.py - 自动对比KAN和传统CNN的性能
import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import subprocess
import threading
import queue
import random
import numpy as np

# 添加父目录到模块路径
sys.path.append(os.path.abspath("E:\GAN_ready\pythonProject"))

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
    print(f"🎯 随机种子已设置为: {seed}")

# 要对比的模型列表 (可以根据需要修改)
MODELS_TO_COMPARE = [
    "lightweight_cnn",    # 轻量级传统CNN
    "lightweight_kan",    # 轻量级KAN CNN
    "traditional_cnn",    # 标准传统CNN
    "residual_cnn",       # 残差传统CNN
    "residual_kan",       # 残差KAN CNN
    "vgg16_lightweight",  # 轻量级VGG16
    "vgg16",              # 标准VGG16
    "deep_kan",           # 深度KAN (召回率优化)
    "ultra_deep_kan",     # 超深度KAN (极致召回率)
    "coatnet",            # CoAtNet混合架构
]

def run_single_experiment(model_type, max_epochs=30, seed=42):
    """运行单个模型实验，显示实时进度"""
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型: {model_type.upper()}")
    print(f"🎯 使用随机种子: {seed}")
    print(f"{'='*60}")
    
    # 使用简化的训练脚本，通过命令行参数传递配置
    script_path = "python_h5/simple_train.py"
    
    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"❌ 训练脚本不存在: {script_path}")
        return None

    try:
        # 使用实时输出的subprocess - 修复缓冲问题
        print(f"📊 {model_type} 训练开始...")
        print("-" * 40)
        
        # 设置环境变量以确保正确的编码和输出
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'  # 关闭Python输出缓冲
        
        # 构建命令行参数
        cmd = [
            sys.executable, '-u', script_path,
            '--model_type', model_type,
            '--max_epochs', str(max_epochs),
            '--seed', str(seed)  # 添加随机种子参数
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,  # 完全无缓冲
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # 实时显示输出 - 使用更简单直接的方法
        epoch_count = 0
        start_time = time.time()
        last_output_time = time.time()
        
        print(f"⏳ [{model_type}] 等待训练开始...")
        
        while True:
            try:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    output = output.strip()
                    current_time = time.time()
                    last_output_time = current_time
                    
                    # 显示所有非空输出（更宽松的过滤）
                    if output and len(output) > 0:
                        # 检查是否包含关键信息
                        is_important = any(keyword in output.lower() for keyword in [
                            'epoch', 'loss', 'acc', 'accuracy', '准确率', '损失', 
                            'training', 'validation', 'val', 'train', 'lr',
                            'best', 'save', 'model', '保存', '模型', 'early', 'stop',
                            'complete', 'finish', '完成', 'error', 'failed', '错误', '失败'
                        ])
                        
                        if is_important:
                            # 检查是否是新的epoch
                            if 'epoch' in output.lower() and ('/' in output or 'of' in output.lower()):
                                epoch_count += 1
                                elapsed_time = current_time - start_time
                                print(f"\n🔄 [{model_type}] Epoch {epoch_count}: {output}")
                                print(f"   ⏱️  已用时: {elapsed_time:.1f}秒")
                            else:
                                print(f"   📊 [{model_type}] {output}")
                        else:
                            # 即使不是重要信息，也偶尔显示以证明程序在运行
                            if current_time - last_output_time > 30:  # 30秒没有重要输出就显示一般输出
                                print(f"   💭 [{model_type}] {output}")
                
                # 检查是否长时间没有输出（可能卡住了）
                if time.time() - last_output_time > 120:  # 2分钟没有输出
                    print(f"⚠️  [{model_type}] 警告: 已有2分钟没有输出，训练可能卡住了...")
                    break
                    
            except UnicodeDecodeError:
                # 如果遇到编码错误，显示但继续
                print(f"   ⚠️  [{model_type}] [编码错误，跳过一行]")
                continue
            except Exception as e:
                print(f"   ❌ [{model_type}] 读取输出时出错: {e}")
                continue
        
        # 等待进程结束
        return_code = process.wait(timeout=3600)
        
        total_time = time.time() - start_time
        print(f"\n🏁 [{model_type}] 训练进程结束")
        print(f"   返回码: {return_code}")
        print(f"   总用时: {total_time:.1f}秒")
        
        if return_code != 0:
            print(f"❌ 模型 {model_type} 训练失败 (返回码: {return_code})")
            return None
        
        # 加载训练结果
        try:
            results_file = f'./training_results_{model_type}.pkl'
            if os.path.exists(results_file):
                results = torch.load(results_file)
                print(f"📈 [{model_type}] 最终准确率: {results['best_val_acc']:.2f}%")
                print(f"⚡ [{model_type}] 参数量: {results['param_count']:,}")
                
                # 显示新增的分类指标
                if 'best_fpr' in results:
                    print(f"🎯 [{model_type}] 最佳假阳率: {results['best_fpr']:.4f}")
                if 'final_f1_score' in results:
                    print(f"📊 [{model_type}] F1分数: {results['final_f1_score']:.4f}")
                
                return results
            else:
                print(f"❌ 结果文件不存在: {results_file}")
                return None
        except Exception as e:
            print(f"❌ 无法加载模型 {model_type} 的训练结果: {e}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 模型 {model_type} 训练超时")
        try:
            process.kill()
        except:
            pass
        return None
    except Exception as e:
        print(f"❌ 运行模型 {model_type} 时出错: {e}")
        return None
    finally:
        print(f"🧹 [{model_type}] 清理完成\n")

def compare_models_interactive():
    """交互式模型对比"""
    print("🚀 KAN vs 传统CNN 对比实验")
    print("-" * 50)
    
    # 让用户选择要对比的模型
    available_models = {
        "1": "lightweight_cnn",
        "2": "lightweight_kan", 
        "3": "traditional_cnn",
        "4": "residual_cnn",
        "5": "residual_kan",
        "6": "improved_kan",
        "7": "vgg16_lightweight",
        "8": "vgg16",
        "9": "deep_kan",
        "10": "ultra_deep_kan",
        "11": "coatnet",
    }
    
    print("可用模型:")
    for key, model in available_models.items():
        print(f"  {key}. {model}")
    
    print("\n请选择要对比的模型 (输入数字，用逗号分隔，例如: 1,2,3):")
    print("💡 建议先选择轻量级模型进行快速测试: 1,2")
    selection = input("您的选择: ").strip()
    
    try:
        selected_indices = [s.strip() for s in selection.split(',')]
        selected_models = [available_models[idx] for idx in selected_indices if idx in available_models]
    except:
        print("输入格式错误，使用默认模型组合...")
        selected_models = ["lightweight_cnn", "lightweight_kan"]
    
    if not selected_models:
        print("未选择有效模型，使用默认组合...")
        selected_models = ["lightweight_cnn", "lightweight_kan"]
    
    print(f"\n✅ 将对比以下模型: {', '.join(selected_models)}")
    
    # 询问训练轮数
    try:
        max_epochs = int(input("\n请输入训练轮数 (建议10-30): ").strip())
        max_epochs = max(1, min(max_epochs, 100))  # 限制在1-100之间
    except:
        max_epochs = 20
        print(f"使用默认训练轮数: {max_epochs}")
    
    # 询问随机种子
    try:
        seed = int(input("\n请输入随机种子 (建议使用42，按回车使用默认): ").strip())
    except:
        seed = 42
        print(f"使用默认随机种子: {seed}")
    
    return selected_models, max_epochs, seed

def generate_comparison_report(results_dict):
    """生成对比报告"""
    if not results_dict:
        print("❌ 没有有效的实验结果")
        return
    
    print(f"\n{'='*80}")
    print("📊 模型对比报告")
    print(f"{'='*80}")
    
    # 创建对比表格
    comparison_data = []
    for model_type, results in results_dict.items():
        if results:
            data_row = {
                '模型类型': model_type,
                '参数量': f"{results['param_count']:,}",
                '训练时间(秒)': f"{results['training_time']:.1f}",
                '最佳验证准确率(%)': f"{results['best_val_acc']:.2f}",
                '是否为KAN': 'KAN' if 'kan' in model_type.lower() else 'CNN'
            }
            
            # 添加新的分类指标
            if 'best_fpr' in results:
                data_row['最佳假阳率'] = f"{results['best_fpr']:.4f}"
            if 'final_f1_score' in results:
                data_row['F1分数'] = f"{results['final_f1_score']:.4f}"
            if 'final_precision' in results:
                data_row['精确率'] = f"{results['final_precision']:.4f}"
            if 'final_recall' in results:
                data_row['召回率'] = f"{results['final_recall']:.4f}"
                
            comparison_data.append(data_row)
    
    if not comparison_data:
        print("❌ 没有有效的实验结果")
        return
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('最佳验证准确率(%)', ascending=False)
    
    print("\n🏆 综合排名:")
    print(df.to_string(index=False))
    
    # 分析结果
    print(f"\n📈 详细分析:")
    
    # 准确率分析
    best_model = df.iloc[0]
    print(f"🥇 最佳模型: {best_model['模型类型']} (准确率: {best_model['最佳验证准确率(%)']}%)")
    
    # 假阳率分析
    if '最佳假阳率' in df.columns:
        df_fpr = df.copy()
        df_fpr['假阳率_数值'] = df_fpr['最佳假阳率'].astype(float)
        best_fpr_model = df_fpr.loc[df_fpr['假阳率_数值'].idxmin()]
        print(f"🎯 最低假阳率: {best_fpr_model['模型类型']} (FPR: {best_fpr_model['最佳假阳率']})")
    
    # KAN vs CNN对比
    kan_models = df[df['是否为KAN'] == 'KAN']
    cnn_models = df[df['是否为KAN'] == 'CNN']
    
    if not kan_models.empty and not cnn_models.empty:
        kan_avg_acc = kan_models['最佳验证准确率(%)'].apply(lambda x: float(x)).mean()
        cnn_avg_acc = cnn_models['最佳验证准确率(%)'].apply(lambda x: float(x)).mean()
        
        print(f"\n🎯 KAN vs CNN 对比:")
        print(f"   • KAN模型平均准确率: {kan_avg_acc:.2f}%")
        print(f"   • CNN模型平均准确率: {cnn_avg_acc:.2f}%")
        
        if kan_avg_acc > cnn_avg_acc:
            print(f"   • 🎊 KAN模型平均比CNN模型高 {kan_avg_acc - cnn_avg_acc:.2f}%")
        else:
            print(f"   • 🎊 CNN模型平均比KAN模型高 {cnn_avg_acc - kan_avg_acc:.2f}%")
        
        # 假阳率对比
        if '最佳假阳率' in df.columns:
            kan_avg_fpr = kan_models['最佳假阳率'].apply(lambda x: float(x)).mean()
            cnn_avg_fpr = cnn_models['最佳假阳率'].apply(lambda x: float(x)).mean()
            print(f"   • KAN模型平均假阳率: {kan_avg_fpr:.4f}")
            print(f"   • CNN模型平均假阳率: {cnn_avg_fpr:.4f}")
    
    # 参数量效率分析
    print(f"\n📊 参数量效率分析:")
    for _, row in df.iterrows():
        param_count = int(row['参数量'].replace(',', ''))
        accuracy = float(row['最佳验证准确率(%)'])
        efficiency = accuracy / (param_count / 1000)  # 每千参数的准确率
        fpr_info = f", FPR: {row.get('最佳假阳率', 'N/A')}" if '最佳假阳率' in row else ""
        print(f"   • {row['模型类型']}: {row['参数量']} 参数 → {accuracy:.2f}% (效率: {efficiency:.3f}{fpr_info})")
    
    # 训练时间分析
    print(f"\n⏱️ 训练时间分析:")
    for _, row in df.iterrows():
        print(f"   • {row['模型类型']}: {row['训练时间(秒)']} 秒")
    
    # 保存详细报告
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"model_comparison_report_{timestamp}.csv"
    try:
        df.to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 详细报告已保存至: {report_file}")
    except Exception as e:
        print(f"\n❌ 保存报告失败: {e}")
    
    # 给出建议
    print(f"\n💡 建议:")
    best_kan = kan_models.iloc[0] if not kan_models.empty else None
    best_cnn = cnn_models.iloc[0] if not cnn_models.empty else None
    
    if best_kan is not None and best_cnn is not None:
        kan_acc = float(best_kan['最佳验证准确率(%)'])
        cnn_acc = float(best_cnn['最佳验证准确率(%)'])
        
        if kan_acc > cnn_acc:
            print(f"   • 对于您的HTRU数据，KAN架构表现更好")
            print(f"   • 推荐使用: {best_kan['模型类型']}")
        else:
            print(f"   • 对于您的HTRU数据，传统CNN表现更好")
            print(f"   • 推荐使用: {best_cnn['模型类型']}")
        
        # 假阳率建议
        if '最佳假阳率' in df.columns:
            print(f"   • 如果对假阳率要求严格，推荐使用: {best_fpr_model['模型类型']}")
            print(f"     (假阳率仅为 {best_fpr_model['最佳假阳率']})")

def main():
    """主函数"""
    print("🔬 深度学习模型对比实验系统")
    print("   KAN CNN vs 传统CNN 性能对比")
    print("   实时显示训练进度")
    print("="*50)
    
    # 交互式选择模型
    selected_models, max_epochs, seed = compare_models_interactive()
    
    # 设置全局随机种子
    set_seed(seed)
    
    print(f"\n🚀 实验配置:")
    print(f"   • 模型数量: {len(selected_models)}")
    print(f"   • 训练轮数: {max_epochs}")
    print(f"   • 随机种子: {seed}")
    print(f"   • 预计时间: {len(selected_models) * max_epochs * 0.5:.1f} 分钟")
    
    input("\n按 Enter 开始实验...")
    
    # 运行实验
    results_dict = {}
    total_start_time = time.time()
    
    for i, model_type in enumerate(selected_models, 1):
        print(f"\n{'🔥' * 20}")
        print(f"进度: {i}/{len(selected_models)} - 当前模型: {model_type}")
        print(f"{'🔥' * 20}")
        
        results = run_single_experiment(model_type, max_epochs, seed)
        results_dict[model_type] = results
        
        if results:
            print(f"✅ {model_type} 训练成功 - 准确率: {results['best_val_acc']:.2f}%")
        else:
            print(f"❌ {model_type} 训练失败")
        
        # 显示剩余时间估计
        if i < len(selected_models):
            elapsed = time.time() - total_start_time
            avg_time_per_model = elapsed / i
            remaining_time = avg_time_per_model * (len(selected_models) - i)
            print(f"⏳ 预计剩余时间: {remaining_time:.1f} 秒")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️ 总实验时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    
    # 生成对比报告
    generate_comparison_report(results_dict)
    
    print(f"\n🎉 实验完成！感谢您的耐心等待!")

if __name__ == "__main__":
    main() 