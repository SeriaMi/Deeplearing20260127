import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule3D(nn.Module):
    """3DASPP的单个分支模块（封装重复逻辑，简化代码）"""
    def __init__(self, in_channels, out_channels, dilation, groups=8):
        super(_ASPPModule3D, self).__init__()
        # 核心逻辑：3D空洞卷积 + GroupNorm + ReLU
        self.atrous_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,          # 固定3×3×3卷积核
            padding=dilation,       # 填充=dilation，保证分辨率不变
            dilation=dilation,      # 扩张率（核心参数）
            bias=False              # 用Norm层，无需偏置
        )
        self.norm = nn.GroupNorm(groups, out_channels)  # 3D数据推荐GroupNorm
        self.relu = nn.ReLU(inplace=True)

        # 初始化卷积层参数（提升训练稳定性）
        nn.init.kaiming_normal_(self.atrous_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """前向传播：卷积 → 归一化 → 激活"""
        x = self.atrous_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ASPP3D(nn.Module):
    """完整的3DASPP模块（核心类）"""
    def __init__(self, in_channels, out_channels=128, dilations=[2, 4, 6], groups=8):
        super(ASPP3D, self).__init__()
        self.out_channels = out_channels
        self.dilations = dilations
        # 每个分支的输出通道数（均分总输出通道，避免通道数爆炸）
        branch_channels = out_channels // (len(dilations) + 2)  # +2：1x1分支 + 全局池化分支
        # 确保branch_channels能被groups整除
        branch_channels = (branch_channels // groups) * groups
        
        # ---------------------- 分支1：1×1×1 3D卷积（局部细节特征） ----------------------
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, branch_channels),
            nn.ReLU(inplace=True)
        )

        # ---------------------- 分支2-4：不同扩张率的3×3×3空洞卷积（多尺度特征） ----------------------
        self.aspp_branches = nn.ModuleList()
        for dilation in dilations:
            self.aspp_branches.append(
                _ASPPModule3D(in_channels, branch_channels, dilation, groups)
            )

        # ---------------------- 分支5：全局平均池化（全局上下文特征） ----------------------
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 自适应池化到1×1×1，捕获全局特征
            nn.Conv3d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, branch_channels),
            nn.ReLU(inplace=True)
        )

        # ---------------------- 特征融合：拼接后用1×1×1卷积调整通道数 ----------------------
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=branch_channels * (len(dilations) + 2),  # 所有分支通道数之和
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5)  # dropout防止过拟合
        )

    def forward(self, x):
        """
        前向传播：并行处理所有分支 → 维度适配 → 拼接 → 融合
        Args:
            x: 输入张量，维度 [B, C, D, H, W]
        Returns:
            融合后的3D特征图，维度 [B, out_channels, D, H, W]
        """
        # 分支1：1x1卷积输出
        x1 = self.conv1x1(x)
        
        # 分支2-4：不同扩张率的空洞卷积输出
        aspp_outs = [x1]  # 先加入1x1分支的输出
        for branch in self.aspp_branches:
            aspp_out = branch(x)
            aspp_outs.append(aspp_out)
        
        # 分支5：全局平均池化输出（需上采样到原分辨率）
        x_global = self.global_avg_pool(x)
        # 上采样：将1×1×1的全局特征扩展到和输入相同的D×H×W
        x_global = F.interpolate(
            x_global,
            size=x.shape[2:],  # 取D、H、W维度
            mode='trilinear',  # 3D数据推荐三线性插值
            align_corners=False
        )
        aspp_outs.append(x_global)

        # 拼接所有分支的输出（按通道维度拼接）
        concat_outs = torch.cat(aspp_outs, dim=1)
        
        # 特征融合：调整通道数并输出
        out = self.fusion_conv(concat_outs)
        
        return out

# ---------------------- 测试3DASPP模块（验证可运行性） ----------------------
if __name__ == "__main__":
    # 1. 创建3DASPP实例
    # 输入通道数64（模拟3D U-Net编码器输出），输出通道数128，扩张率[2,4,6]
    aspp3d = ASPP3D(in_channels=64, out_channels=128, dilations=[2,4,6])
    
    # 2. 构造模拟输入（医学影像常见维度：B=2, C=64, D=32, H=32, W=32）
    # B=批次大小，C=通道数，D=深度，H=高度，W=宽度
    batch_input = torch.randn(2, 64, 32, 32, 32)
    
    # 3. 前向传播
    with torch.no_grad():  # 无梯度计算，加快速度
        batch_output = aspp3d(batch_input)
    
    # 4. 打印输入输出维度（验证分辨率不变，通道数符合预期）
    print("="*50)
    print(f"输入维度: {batch_input.shape}")  # 预期：torch.Size([2, 64, 32, 32, 32])
    print(f"输出维度: {batch_output.shape}")  # 预期：torch.Size([2, 128, 32, 32, 32])
    print("="*50)
    print("3DASPP模块运行成功！输出分辨率与输入一致，通道数调整为128。")