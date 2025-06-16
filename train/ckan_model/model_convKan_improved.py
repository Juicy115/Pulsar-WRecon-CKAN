import KANConv
import torch
import torch.nn as nn
import torch.nn.functional as F

KAN_Convolutional_Layer = KANConv.KAN_Convolutional_Layer

class ImprovedKANC_MLP(nn.Module):
    def __init__(self, device: str = 'cpu', num_classes: int = 2, dropout_rate: float = 0.5, input_channels: int = 3):
        super().__init__()
        
        # 计算每层的输出通道数: input_channels * n_convs
        conv1_out_channels = input_channels * 3  # 3 * 3 = 9
        conv2_out_channels = conv1_out_channels * 4  # 9 * 4 = 36  
        conv3_out_channels = conv2_out_channels * 2  # 36 * 2 = 72
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=3, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(conv1_out_channels),  # 9个通道
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=4, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(conv2_out_channels),  # 36个通道
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # 第三个卷积块 (减少n_convs避免通道数过多)
        self.conv_block3 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(conv3_out_channels),  # 72个通道
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化到4x4
        )
        
        # 计算展平后的特征数: 72 * 4 * 4 = 1152
        flatten_features = conv3_out_channels * 4 * 4
        self.flatten = nn.Flatten()
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(flatten_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 展平和分类
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x  # 直接返回logits，让损失函数处理softmax


class ResidualKANC_MLP(nn.Module):
    """带残差连接的CNNKAN网络"""
    def __init__(self, device: str = 'cpu', num_classes: int = 2, dropout_rate: float = 0.3, input_channels: int = 3):
        super().__init__()
        
        # 正确计算KAN卷积层的输出通道数
        # KAN层输出通道数 = 输入通道数 × n_convs
        conv1_out = input_channels * 2   # 3 × 2 = 6
        conv2_out = conv1_out * 2        # 6 × 2 = 12
        conv3_out = conv2_out * 3        # 12 × 3 = 36
        
        # 第一个卷积块
        self.conv1 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn1 = nn.BatchNorm2d(conv1_out)  # 6个通道
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 第二个卷积块
        self.conv2 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn2 = nn.BatchNorm2d(conv2_out)  # 12个通道
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 第三个卷积块
        self.conv3 = KAN_Convolutional_Layer(n_convs=3, kernel_size=(3, 3), device=device)
        self.bn3 = nn.BatchNorm2d(conv3_out)  # 36个通道
        
        # 残差连接的投影层 (从12通道投影到36通道)
        self.proj_conv = nn.Conv2d(conv2_out, conv3_out, kernel_size=1)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器 (更新flatten_features计算)
        flatten_features = conv3_out * 4 * 4  # 36 * 4 * 4 = 576
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 第一个卷积块
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x2 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
        
        # 第三个卷积块带残差连接
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # 残差连接 (需要调整x2的通道数)
        x2_projected = self.proj_conv(x2)
        if x3.shape[2:] != x2_projected.shape[2:]:
            # 如果空间维度不匹配，使用插值调整
            x2_projected = F.interpolate(x2_projected, size=x3.shape[2:], mode='bilinear', align_corners=False)
        
        x3 = x3 + x2_projected  # 残差连接
        
        # 池化和分类
        x = self.adaptive_pool(x3)
        x = self.classifier(x)
        
        return x


# 轻量级版本，适合快速实验
class LightweightKANC_MLP(nn.Module):
    def __init__(self, device: str = 'cpu', num_classes: int = 2, input_channels: int = 3):
        super().__init__()
        
        # 计算输出通道数
        conv1_out = input_channels * 2  # 3 * 2 = 6
        conv2_out = conv1_out * 2       # 6 * 2 = 12
        
        self.features = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(conv1_out),
            nn.MaxPool2d(2),
            
            KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(conv2_out),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        flatten_features = conv2_out * 4 * 4  # 12 * 4 * 4 = 192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x 


# ========================= 传统CNN模型（用于对比实验） =========================

class TraditionalCNN(nn.Module):
    """传统CNN模型，架构与ImprovedKANC_MLP对应"""
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, input_channels: int = 3):
        super().__init__()
        
        # 与KAN CNN相似的通道数设计
        conv1_out = 16
        conv2_out = 32
        conv3_out = 64
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_out),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 分类器部分
        flatten_features = conv3_out * 4 * 4  # 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


class ResidualCNN(nn.Module):
    """带残差连接的传统CNN"""
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3, input_channels: int = 3):
        super().__init__()
        
        conv_channels = 32
        final_channels = 64
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(conv_channels, final_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(final_channels)
        
        # 残差连接的投影层
        self.proj_conv = nn.Conv2d(conv_channels, final_channels, kernel_size=1)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器
        flatten_features = final_channels * 4 * 4  # 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一个卷积块
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x2 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
        
        # 第三个卷积块带残差连接
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # 残差连接
        x2_projected = self.proj_conv(x2)
        if x3.shape[2:] != x2_projected.shape[2:]:
            x2_projected = F.interpolate(x2_projected, size=x3.shape[2:], mode='bilinear', align_corners=False)
        
        x3 = x3 + x2_projected
        
        # 池化和分类
        x = self.adaptive_pool(x3)
        x = self.classifier(x)
        return x


class LightweightCNN(nn.Module):
    """轻量级传统CNN"""
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x 


class VGG16(nn.Module):
    """VGG16网络，专门为32x32输入图像优化"""
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, input_channels: int = 3):
        super().__init__()
        
        # VGG16特征提取层配置 (针对32x32输入优化)
        # 原始VGG16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # 针对32x32优化的配置 (减少层数避免特征图过小)
        self.cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M']
        
        self.features = self._make_layers(input_channels)
        
        # 自适应池化确保固定输出尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 分类器 (针对小输入图像调整)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024),  # 256个通道 * 2x2
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layers(self, input_channels):
        """构建VGG特征提取层"""
        layers = []
        in_channels = input_channels
        
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 自适应池化
        x = self.adaptive_pool(x)
        # 分类
        x = self.classifier(x)
        return x


class VGG16_Lightweight(nn.Module):
    """轻量级VGG16，适合快速实验"""
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3, input_channels: int = 3):
        super().__init__()
        
        # 更轻量的VGG配置
        self.cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M']
        
        self.features = self._make_layers(input_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 轻量级分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layers(self, input_channels):
        layers = []
        in_channels = input_channels
        
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x 


class DeepKANC_MLP(nn.Module):
    """深度KAN网络，专门为提升召回率优化
    - 更多的KAN卷积层（5层）
    - 多个残差连接
    - 专门针对召回率优化的架构
    - 渐进式特征提取
    """
    def __init__(self, device: str = 'cpu', num_classes: int = 2, dropout_rate: float = 0.2, input_channels: int = 3):
        super().__init__()
        
        # 深度KAN网络的通道配置（渐进式增长）
        conv1_out = input_channels * 2   # 3 × 2 = 6
        conv2_out = conv1_out * 2        # 6 × 2 = 12  
        conv3_out = conv2_out * 2        # 12 × 2 = 24
        conv4_out = conv3_out * 2        # 24 × 2 = 48
        conv5_out = conv4_out * 1        # 48 × 1 = 48 (保持)
        
        # 第一个KAN卷积块
        self.conv1 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 第二个KAN卷积块
        self.conv2 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn2 = nn.BatchNorm2d(conv2_out)
        
        # 第三个KAN卷积块（第一个残差连接）
        self.conv3 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn3 = nn.BatchNorm2d(conv3_out)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # 残差投影层1 (6→24)
        self.proj_conv1 = nn.Conv2d(conv1_out, conv3_out, kernel_size=1)
        
        # 第四个KAN卷积块
        self.conv4 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn4 = nn.BatchNorm2d(conv4_out)
        
        # 第五个KAN卷积块（第二个残差连接）
        self.conv5 = KAN_Convolutional_Layer(n_convs=1, kernel_size=(3, 3), device=device)
        self.bn5 = nn.BatchNorm2d(conv5_out)
        
        # 残差投影层2 (24→48)
        self.proj_conv2 = nn.Conv2d(conv3_out, conv5_out, kernel_size=1)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 针对召回率优化的分类器
        flatten_features = conv5_out * 4 * 4  # 48 * 4 * 4 = 768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 更大的隐藏层提升特征表达能力
            nn.Linear(flatten_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 第二个隐藏层
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 第三个隐藏层（有助于复杂特征学习）
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),  # 较低的dropout防止欠拟合
            
            # 输出层
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """特殊的权重初始化，有助于提升召回率"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用稍大的权重初始化，有利于召回率
                nn.init.xavier_uniform_(m.weight, gain=1.2)
                if m.bias is not None:
                    # 输出层偏置设置为略负，降低决策阈值
                    if m.out_features == 2:  # 输出层
                        nn.init.constant_(m.bias, 0)
                        m.bias.data[0] = -0.1  # 负类偏置
                        m.bias.data[1] = 0.1   # 正类偏置
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一个KAN卷积块
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))  # 16x16
        
        # 第二个KAN卷积块
        x2 = F.relu(self.bn2(self.conv2(x1)))  # 16x16
        
        # 第三个KAN卷积块 + 第一个残差连接
        x3 = self.pool2(F.relu(self.bn3(self.conv3(x2))))  # 8x8
        
        # 残差连接1：x1 -> x3
        x1_projected = self.proj_conv1(x1)  # 投影到24通道
        x1_projected = F.adaptive_avg_pool2d(x1_projected, x3.shape[2:])  # 空间降采样到匹配x3
        x3 = x3 + x1_projected
        
        # 第四个KAN卷积块
        x4 = F.relu(self.bn4(self.conv4(x3)))  # 8x8
        
        # 第五个KAN卷积块 + 第二个残差连接
        x5 = F.relu(self.bn5(self.conv5(x4)))  # 8x8
        
        # 残差连接2：x3 -> x5
        x3_projected = self.proj_conv2(x3)  # 投影到48通道
        # 确保空间维度匹配
        if x5.shape[2:] != x3_projected.shape[2:]:
            x3_projected = F.adaptive_avg_pool2d(x3_projected, x5.shape[2:])
        x5 = x5 + x3_projected
        
        # 最终池化和分类
        x = self.adaptive_pool(x5)  # 4x4
        x = self.classifier(x)
        
        return x


class UltraDeepKANC_MLP(nn.Module):
    """超深度KAN网络，极致召回率优化
    简化但可靠的版本
    """
    def __init__(self, device: str = 'cpu', num_classes: int = 2, dropout_rate: float = 0.15, input_channels: int = 3):
        super().__init__()
        
        # 明确的通道数计算，确保正确
        conv1_out = input_channels * 2  # 3 × 2 = 6
        conv2_out = conv1_out * 2       # 6 × 2 = 12
        conv3_out = conv2_out * 2       # 12 × 2 = 24
        conv4_out = conv3_out * 2       # 24 × 2 = 48
        conv5_out = conv4_out * 1       # 48 × 1 = 48
        conv6_out = conv5_out * 1       # 48 × 1 = 48
        
        # 第1个KAN卷积块
        self.conv1 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.pool1 = nn.MaxPool2d(2)
        
        # 第2个KAN卷积块
        self.conv2 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn2 = nn.BatchNorm2d(conv2_out)
        
        # 第3个KAN卷积块
        self.conv3 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn3 = nn.BatchNorm2d(conv3_out)
        self.pool2 = nn.MaxPool2d(2)
        
        # 第4个KAN卷积块
        self.conv4 = KAN_Convolutional_Layer(n_convs=2, kernel_size=(3, 3), device=device)
        self.bn4 = nn.BatchNorm2d(conv4_out)
        
        # 第5个KAN卷积块
        self.conv5 = KAN_Convolutional_Layer(n_convs=1, kernel_size=(3, 3), device=device)
        self.bn5 = nn.BatchNorm2d(conv5_out)
        
        # 第6个KAN卷积块
        self.conv6 = KAN_Convolutional_Layer(n_convs=1, kernel_size=(3, 3), device=device)
        self.bn6 = nn.BatchNorm2d(conv6_out)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 召回率优化分类器
        flatten_features = conv6_out * 4 * 4  # 48 * 4 * 4 = 768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """召回率优化的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.3)
                if m.bias is not None:
                    if m.out_features == 2:
                        nn.init.constant_(m.bias, 0)
                        m.bias.data[0] = -0.2
                        m.bias.data[1] = 0.2
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第1层：32x32 -> 16x16
        x = F.relu(self.bn1(self.conv1(x)))  # 6通道
        x = self.pool1(x)  # 16x16
        
        # 第2层：16x16
        x = F.relu(self.bn2(self.conv2(x)))  # 12通道
        
        # 第3层：16x16 -> 8x8
        x = F.relu(self.bn3(self.conv3(x)))  # 24通道
        x = self.pool2(x)  # 8x8
        
        # 第4层：8x8
        x = F.relu(self.bn4(self.conv4(x)))  # 48通道
        
        # 第5层：8x8
        x = F.relu(self.bn5(self.conv5(x)))  # 48通道
        
        # 第6层：8x8
        x = F.relu(self.bn6(self.conv6(x)))  # 48通道
        
        # 最终池化和分类
        x = self.adaptive_pool(x)  # 4x4
        x = self.classifier(x)
        
        return x 


# ========================= CoAtNet 混合架构 =========================

class MBConv(nn.Module):
    """MobileNet风格的卷积块，用于CoAtNet的卷积阶段"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_channels = in_channels * expand_ratio
        
        # 扩展卷积
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        ) if expand_ratio != 1 else nn.Identity()
        
        # 深度卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=stride, 
                     padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        # SE注意力机制
        self.se = SEBlock(hidden_channels, int(hidden_channels * se_ratio))
        
        # 投影卷积
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        shortcut = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x = x + shortcut
            
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels, reduction_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduction_channels, 1),
            nn.GELU(),
            nn.Conv2d(reduction_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class RelativePositionBias(nn.Module):
    """相对位置偏置，用于Transformer阶段"""
    def __init__(self, num_heads, max_distance=7):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # 相对位置偏置表
        num_relative_distance = (2 * max_distance + 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_distance, num_heads)
        )
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, height, width):
        # 生成相对位置索引
        coords_h = torch.arange(height, dtype=torch.long)
        coords_w = torch.arange(width, dtype=torch.long)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += height - 1
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        
        relative_position_index = relative_coords.sum(-1)
        
        # 限制在最大距离内并确保数据类型正确
        relative_position_index = torch.clamp(relative_position_index, 
                                            0, len(self.relative_position_bias_table) - 1).long()
        
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(height * width, height * width, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        return relative_position_bias


class TransformerBlock(nn.Module):
    """Transformer块，用于CoAtNet的Transformer阶段"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head Self-Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 相对位置偏置
        self.relative_position_bias = RelativePositionBias(num_heads)
        
    def forward(self, x, height, width):
        B, N, C = x.shape
        
        # Self-Attention
        shortcut = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias(height, width)
        if relative_position_bias.device != attn.device:
            relative_position_bias = relative_position_bias.to(attn.device)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = shortcut + x
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x


class CoAtNet(nn.Module):
    """CoAtNet: 结合卷积和Transformer的混合架构
    专门为32x32 HTRU数据设计
    """
    def __init__(self, num_classes=2, num_blocks=[2, 2, 3, 5], channels=[64, 96, 192, 384], 
                 num_heads=[0, 0, 6, 12], dropout=0.1):
        super().__init__()
        
        self.num_stages = len(num_blocks)
        assert len(channels) == self.num_stages
        assert len(num_heads) == self.num_stages
        
        # Stem: 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(channels[0]),
            nn.GELU()
        )
        
        # 构建各个阶段
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            stage_blocks = []
            in_channels = channels[i-1] if i > 0 else channels[0]
            out_channels = channels[i]
            
            # 下采样（除了第一个阶段）
            if i > 0:
                if num_heads[i] == 0:  # 卷积阶段
                    stage_blocks.append(MBConv(in_channels, out_channels, stride=2))
                else:  # Transformer阶段
                    # 使用卷积进行下采样，然后转换为Transformer
                    stage_blocks.append(MBConv(in_channels, out_channels, stride=2))
            else:
                if in_channels != out_channels:
                    stage_blocks.append(MBConv(in_channels, out_channels, stride=1))
            
            # 添加多个块
            for j in range(num_blocks[i]):
                if num_heads[i] == 0:  # 卷积阶段
                    stage_blocks.append(MBConv(out_channels, out_channels, stride=1))
                else:  # Transformer阶段
                    stage_blocks.append(TransformerBlock(out_channels, num_heads[i], dropout=dropout))
            
            self.stages.append(nn.ModuleList(stage_blocks))
        
        # 分类头
        self.norm = nn.LayerNorm(channels[-1])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)  # [B, 64, 16, 16]
        
        # 各个阶段
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                if isinstance(block, TransformerBlock):
                    # Transformer块需要特殊处理
                    B, C, H, W = x.shape
                    x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                    x = block(x, H, W)
                    x = x.transpose(1, 2).reshape(B, C, H, W)
                else:
                    # 卷积块
                    x = block(x)
        
        # 分类头
        if len(x.shape) == 3:  # 如果最后是Transformer输出
            x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], 
                                        int(x.shape[1]**0.5), int(x.shape[1]**0.5))
        
        x = self.head(x)
        return x


class CoAtNet_Simple(nn.Module):
    """CoAtNet简化版本，更稳定可靠"""
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        
        # Stem卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # 卷积阶段1: 16x16
        self.conv_stage1 = nn.Sequential(
            MBConv(64, 96, stride=1),
            MBConv(96, 96, stride=1)
        )
        
        # 卷积阶段2: 8x8
        self.conv_stage2 = nn.Sequential(
            MBConv(96, 192, stride=2),
            MBConv(192, 192, stride=1)
        )
        
        # 简化的自注意力块（不使用复杂的相对位置偏置）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=192,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(192)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(192, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # 卷积阶段
        x = self.stem(x)        # [B, 64, 16, 16]
        x = self.conv_stage1(x) # [B, 96, 16, 16]
        x = self.conv_stage2(x) # [B, 192, 8, 8]
        
        # 转换为序列进行自注意力
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 192]
        
        # 自注意力
        shortcut = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x)
        x = shortcut + x
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        # 转换回图像格式并分类
        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, 192, 8, 8]
        x = self.head(x)
        
        return x 