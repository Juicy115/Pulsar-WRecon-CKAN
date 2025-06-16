# -*- coding: utf-8 -*-
"""
简化训练脚本 - 通过命令行参数接收配置
避免动态修改脚本带来的缩进问题

🎯 随机种子控制说明 (论文级别的可重复性保证):
=======================================================

本脚本已实现全面的随机种子控制，确保实验的完全可重复性，满足学术论文发表要求：

1. 🎲 全局随机种子设置:
   - Python random模块
   - NumPy随机数生成器  
   - PyTorch CPU随机数生成器
   - PyTorch CUDA随机数生成器
   - CUDNN确定性模式

2. 📊 数据相关随机性控制:
   - 数据集分割 (train_test_split)
   - 数据增强 (RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter)
   - 数据采样 (balanced_subsample)
   - DataLoader批次顺序 (shuffle=True with generator)
   - 多进程数据加载 (worker_init_fn)

3. 🧠 模型相关随机性控制:
   - 权重初始化 (xavier_uniform, kaiming_normal等)
   - Dropout层随机性
   - 模型架构中的随机组件

4. 🔧 训练相关随机性控制:
   - 优化器初始化
   - 学习率调度器
   - 每个epoch的随机性
   - 梯度计算中的随机性

5. 🎯 种子分配策略:
   - 主种子: args.seed
   - 数据增强: args.seed + 1
   - 训练采样: args.seed + 2  
   - 验证采样: args.seed + 3
   - 训练DataLoader: args.seed + 4
   - 验证DataLoader: args.seed + 5
   - 权重初始化: args.seed + 100
   - 每个epoch: args.seed + epoch * 1000

使用方法:
python simple_train.py --model_type lightweight_kan --max_epochs 30 --seed 42

这确保了:
✅ 完全的实验可重复性
✅ 公平的模型对比
✅ 符合学术论文标准
✅ 便于调试和验证
"""

import sys
import os
import argparse

# 添加父目录到模块路径
sys.path.append(os.path.abspath("E:\GAN_ready\pythonProject"))
import random
from model_convKan_improved import (
    ImprovedKANC_MLP, ResidualKANC_MLP, LightweightKANC_MLP,
    TraditionalCNN, ResidualCNN, LightweightCNN, VGG16, VGG16_Lightweight,
    DeepKANC_MLP, UltraDeepKANC_MLP, CoAtNet_Simple
)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from htru1 import HTRU1
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time

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

def worker_init_fn(worker_id):
    """DataLoader工作进程的随机种子初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 修复Windows多进程问题
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练指定的模型')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['improved_kan', 'residual_kan', 'lightweight_kan',
                               'traditional_cnn', 'residual_cnn', 'lightweight_cnn', 'vgg16', 'vgg16_lightweight',
                               'deep_kan', 'ultra_deep_kan', 'coatnet', 'coatnet_small'],
                       help='模型类型')
    parser.add_argument('--max_epochs', type=int, default=30, help='最大训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"选择的模型: {args.model_type}")
    print(f"训练轮数: {args.max_epochs}")
    print(f"随机种子: {args.seed}")

    # 改进的数据增强策略 - 使用固定的随机种子
    # 创建一个生成器来控制数据增强的随机性
    def create_transform_with_seed(seed_offset=0):
        """创建带有固定随机种子的数据变换"""
        # 为数据增强设置独立的随机种子
        transform_seed = args.seed + seed_offset
        torch.manual_seed(transform_seed)
        np.random.seed(transform_seed)
        random.seed(transform_seed)
        
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_train = create_transform_with_seed(seed_offset=1)

    # 测试时不做数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 重新设置随机种子，确保后续操作的一致性
    set_seed(args.seed)

    # 加载完整数据集（不使用预定义的train/test分割）
    print("🔄 加载并重新分割数据集...")
    
    # 先加载所有数据（使用train=True获取更多数据，然后自己分割）
    full_dataset = HTRU1('data', train=True, download=True, transform=None)  # 先不应用transform
    
    # 获取所有样本的索引和标签
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset.targets[i] for i in all_indices]
    
    print(f"📊 原始数据集统计:")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = "脉冲星" if label == 1 else "非脉冲星"
        print(f"   {label_name} (类别{label}): {count} 个样本")
    
    # 分层分割：80%训练，20%验证（保持类别比例）
    train_indices, val_indices = train_test_split(
        all_indices, 
        test_size=0.2, 
        random_state=args.seed,  # 使用传入的随机种子确保可重复性
        stratify=all_labels  # 分层抽样保持类别比例
    )
    
    print(f"\n🔄 数据分割结果:")
    print(f"   训练集样本数: {len(train_indices)}")
    print(f"   验证集样本数: {len(val_indices)}")
    
    # 检查分割后的类别分布
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    
    print(f"\n📊 训练集类别分布:")
    for label, count in zip(train_unique, train_counts):
        label_name = "脉冲星" if label == 1 else "非脉冲星"
        percentage = count / len(train_labels) * 100
        print(f"   {label_name}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n📊 验证集类别分布:")
    for label, count in zip(val_unique, val_counts):
        label_name = "脉冲星" if label == 1 else "非脉冲星"
        percentage = count / len(val_labels) * 100
        print(f"   {label_name}: {count} 个样本 ({percentage:.1f}%)")

    # 进一步采样以平衡训练数据（可选）
    def balanced_subsample(indices, labels, pos_samples=1000, neg_samples=1000, seed_offset=0):
        """从给定索引中平衡采样"""
        # 为采样设置独立的随机种子
        sampling_seed = args.seed + seed_offset
        random.seed(sampling_seed)
        np.random.seed(sampling_seed)
        
        # 获取正负样本索引
        pos_indices = [idx for idx in indices if all_labels[idx] == 1]  # 脉冲星
        neg_indices = [idx for idx in indices if all_labels[idx] == 0]  # 非脉冲星
        
        # 确保不超过可用数量
        pos_samples = min(pos_samples, len(pos_indices))
        neg_samples = min(neg_samples, len(neg_indices))
        
        # 随机采样
        selected_pos = random.sample(pos_indices, pos_samples)
        selected_neg = random.sample(neg_indices, neg_samples)
        
        selected_indices = selected_pos + selected_neg
        random.shuffle(selected_indices)  # 打乱顺序
        
        print(f"   采样结果: {pos_samples} 个脉冲星 + {neg_samples} 个非脉冲星 = {len(selected_indices)} 个样本")
        return selected_indices

    # 对训练集进行平衡采样（根据需要调整样本数）
    train_sampled_indices = balanced_subsample(train_indices, train_labels, 
                                             pos_samples=1000, neg_samples=2000, seed_offset=2)
    
    # 对验证集也可以进行采样（保持更平衡的评估）
    val_sampled_indices = balanced_subsample(val_indices, val_labels,
                                           pos_samples=200, neg_samples=1800, seed_offset=3)
    
    print(f"\n✅ 最终数据集:")
    print(f"   训练集: {len(train_sampled_indices)} 个样本")
    print(f"   验证集: {len(val_sampled_indices)} 个样本")
    
    # 创建专门的数据集类来应用不同的transforms
    class IndexedDataset:
        def __init__(self, base_dataset, indices, transform=None):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.base_dataset[real_idx]
            
            if self.transform:
                image = self.transform(image)
            else:
                # 默认转换
                if not isinstance(image, torch.Tensor):
                    image = transforms.ToTensor()(image)
                    
            return image, label

    # 创建训练和验证数据集
    train_dataset = IndexedDataset(full_dataset, train_sampled_indices, transform_train)
    val_dataset = IndexedDataset(full_dataset, val_sampled_indices, transform_test)

    print(f"训练数据集长度: {len(train_dataset)}")
    print(f"验证数据集长度: {len(val_dataset)}")

    # 数据加载器 - 修复Windows多进程问题，设置num_workers=0，并添加随机种子控制
    # 创建自定义的生成器来控制DataLoader的随机性
    def create_dataloader_generator(seed_offset=0):
        """创建DataLoader使用的随机数生成器"""
        g = torch.Generator()
        g.manual_seed(args.seed + seed_offset)
        return g

    train_generator = create_dataloader_generator(seed_offset=4)
    val_generator = create_dataloader_generator(seed_offset=5)

    trainloader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=0,
        generator=train_generator,
        worker_init_fn=worker_init_fn
    )
    testloader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=0,
        generator=val_generator,
        worker_init_fn=worker_init_fn
    )

    # 重新设置随机种子，确保模型初始化的一致性
    set_seed(args.seed)

    # 模型实例化
    def create_model(model_type):
        if model_type == "improved_kan":
            return ImprovedKANC_MLP(device=device, num_classes=2, dropout_rate=0.5)
        elif model_type == "residual_kan":
            return ResidualKANC_MLP(device=device, num_classes=2, dropout_rate=0.3)
        elif model_type == "lightweight_kan":
            return LightweightKANC_MLP(device=device, num_classes=2)
        elif model_type == "traditional_cnn":
            return TraditionalCNN(num_classes=2, dropout_rate=0.5)
        elif model_type == "residual_cnn":
            return ResidualCNN(num_classes=2, dropout_rate=0.3)
        elif model_type == "lightweight_cnn":
            return LightweightCNN(num_classes=2)
        elif model_type == "vgg16":
            return VGG16(num_classes=2, dropout_rate=0.5)
        elif model_type == "vgg16_lightweight":
            return VGG16_Lightweight(num_classes=2)
        elif model_type == "deep_kan":
            return DeepKANC_MLP(device=device, num_classes=2, dropout_rate=0.5)
        elif model_type == "ultra_deep_kan":
            return UltraDeepKANC_MLP(device=device, num_classes=2, dropout_rate=0.5)
        elif model_type == "coatnet":
            return CoAtNet_Simple(num_classes=2, dropout=0.1)
        elif model_type == "coatnet_small":
            return CoAtNet_Simple(num_classes=2, dropout=0.1)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    model = create_model(args.model_type)

    # 确保模型权重初始化的随机性受控
    # 对于有自定义初始化的模型，重新应用初始化
    if hasattr(model, '_initialize_weights'):
        # 重新设置种子后重新初始化权重
        set_seed(args.seed + 100)  # 使用稍微不同的种子避免与其他组件冲突
        model._initialize_weights()
        print("🎯 模型权重已重新初始化（受随机种子控制）")

    # 计算模型参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_count = count_parameters(model)
    print(f"模型参数量: {param_count:,}")

    # 使用DataParallel（如果有多个GPU）
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)

    model = model.to(device)

    # 计算类别权重（处理类别不平衡）
    def calculate_class_weights(dataset):
        labels = []
        # 根据新的数据集结构获取标签
        if hasattr(dataset, 'indices'):
            # 对于IndexedDataset
            for idx in dataset.indices:
                labels.append(full_dataset.targets[idx])
        else:
            # 对于传统数据集
            for _, label in dataset:
                labels.append(label)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = [total / (len(unique_labels) * count) for count in counts]
        return torch.FloatTensor(weights)

    class_weights = calculate_class_weights(train_dataset).to(device)
    print(f"类别权重: {class_weights}")
    
    # 针对深度KAN模型的召回率优化配置
    if 'deep_kan' in args.model_type or 'ultra_deep' in args.model_type:
        # 为召回率优化调整类别权重 - 增加正类权重
        if len(class_weights) == 2:
            # 增强正类权重以提升召回率
            enhanced_weights = class_weights.clone()
            enhanced_weights[1] = enhanced_weights[1] * 1.5  # 正类权重增加50%
            class_weights = enhanced_weights
            print(f"📈 召回率优化 - 调整后类别权重: {class_weights}")
            
        # 使用Focal Loss来进一步优化召回率
        from torch.nn import CrossEntropyLoss
        
        class FocalLoss(nn.Module):
            """Focal Loss专门用于提升召回率"""
            def __init__(self, alpha=0.75, gamma=2.0, weight=None):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.weight = weight
                self.ce_loss = CrossEntropyLoss(weight=weight, reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                
                # 对正类给予更多关注
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
                
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=0.8, gamma=2.0, weight=class_weights)  # 更关注正类
        print("📈 使用Focal Loss优化召回率")
    else:
        # 损失函数
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 验证一下数据集确实分离
    train_set = set(train_sampled_indices)
    val_set = set(val_sampled_indices)
    overlap = train_set & val_set
    print(f"\n✅ 数据分离验证:")
    print(f"   训练集和验证集重叠样本数: {len(overlap)}")
    if len(overlap) == 0:
        print("   ✅ 数据集完全分离，无数据泄露风险")
    else:
        print("   ❌ 警告：存在数据泄露风险！")

    # 重新设置随机种子，确保优化器初始化的一致性
    set_seed(args.seed)

    # 优化器（针对深度KAN模型调整学习率）
    if 'deep_kan' in args.model_type or 'ultra_deep' in args.model_type:
        # 深度模型使用稍低的学习率
        initial_lr = 5e-4
        print(f"📈 深度KAN模型使用较低学习率: {initial_lr}")
    else:
        initial_lr = 1e-3
    
    # 创建优化器时确保随机性控制
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4, betas=(0.9, 0.999))

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    # 再次设置随机种子，确保训练过程的一致性
    set_seed(args.seed)
    
    print(f"\n🎯 所有随机性组件已设置完成，使用种子: {args.seed}")
    print("   ✅ 数据增强随机性已控制")
    print("   ✅ 数据采样随机性已控制") 
    print("   ✅ DataLoader随机性已控制")
    print("   ✅ 模型初始化随机性已控制")
    print("   ✅ 优化器初始化随机性已控制")
    print("   ✅ 训练过程随机性已控制")

    # 早停机制
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float('inf')
        
        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    early_stopping = EarlyStopping(patience=10)

    # 训练函数
    def train_epoch(model, trainloader, criterion, optimizer, device):
        # 确保模型处于训练模式，Dropout等层会正常工作
        model.train()
        # 确保Dropout的随机性是确定的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(trainloader, desc="Training") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        return running_loss / len(trainloader), 100.0 * correct / total

    # 验证函数（在独立的验证集上评估）
    def validate_epoch(model, valloader, criterion, device):
        # 确保模型处于评估模式，Dropout等层会被禁用
        model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(valloader, desc="Validation") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
        
        # 计算假阳率和其他分类指标
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 假设标签: 0=负类(非脉冲星), 1=正类(脉冲星)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            false_positive_rate = 0.0
            false_negative_rate = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        
        accuracy = 100.0 * correct / total
        
        return (running_loss / len(valloader), accuracy, all_predictions, all_labels, 
                false_positive_rate, false_negative_rate, precision, recall, f1_score)

    # 主训练循环
    best_val_acc = 0.0
    best_fpr = float('inf')  # 最佳假阳率（越低越好）
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    fprs, fnrs, precisions, recalls, f1_scores = [], [], [], [], []  # 新增指标记录
    
    print(f"\n开始训练 {args.model_type} 模型...")
    print(f"参数量: {param_count:,}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        # 为每个epoch设置确定性的随机种子
        epoch_seed = args.seed + epoch * 1000
        set_seed(epoch_seed)
        
        print(f"\nEpoch {epoch+1}/{args.max_epochs} (种子: {epoch_seed})")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # 验证（注意：这里使用的是独立的验证集，与训练集完全分离）
        val_loss, val_acc, predictions, true_labels, false_positive_rate, false_negative_rate, precision, recall, f1_score = validate_epoch(model, testloader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        fprs.append(false_positive_rate)
        fnrs.append(false_negative_rate)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"假阳率(FPR): {false_positive_rate:.4f}, 假阴率(FNR): {false_negative_rate:.4f}")
        print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}")
        
        # 保存最佳模型（基于准确率，但也考虑假阳率）
        if val_acc > best_val_acc or (val_acc == best_val_acc and false_positive_rate < best_fpr):
            best_val_acc = val_acc
            best_fpr = false_positive_rate
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f'./best_model_{args.model_type}.pth')
            else:
                torch.save(model.state_dict(), f'./best_model_{args.model_type}.pth')
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%, 假阳率: {best_fpr:.4f}")
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"早停触发，在第{epoch+1}轮停止训练")
            break
        
        # 每10轮打印详细分类报告
        if (epoch + 1) % 10 == 0:
            print("\n分类报告:")
            print(classification_report(true_labels, predictions, 
                                      target_names=['Non-Pulsar', 'Pulsar']))
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 最终结果总结
    print("\n" + "="*80)
    print(f"训练完成！")
    print(f"模型类型: {args.model_type}")
    print(f"参数量: {param_count:,}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳假阳率: {best_fpr:.4f}")
    print("="*80)
    
    # 保存训练历史（包含所有新指标）
    results = {
        'model_type': args.model_type,
        'param_count': param_count,
        'training_time': training_time,
        'best_val_acc': best_val_acc,
        'best_fpr': best_fpr,
        'final_fpr': fprs[-1] if fprs else 0.0,
        'final_fnr': fnrs[-1] if fnrs else 0.0,
        'final_precision': precisions[-1] if precisions else 0.0,
        'final_recall': recalls[-1] if recalls else 0.0,
        'final_f1_score': f1_scores[-1] if f1_scores else 0.0,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'fprs': fprs,
        'fnrs': fnrs,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores
    }
    
    torch.save(results, f'./training_results_{args.model_type}.pkl')

if __name__ == '__main__':
    main() 