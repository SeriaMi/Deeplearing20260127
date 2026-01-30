import torch
import torch.nn as nn

# 1. 模拟CNN卷积层输出：batch=2（2个样本），channel=1（单通道），H=28，W=28
# 张量形状：[2, 1, 28, 28]
x = torch.randint(0, 5, size=(2, 1, 28, 28))
print("原始张量形状：", x.shape)  # 输出：torch.Size([2, 1, 28, 28])

# 2. 定义Flatten层（使用默认参数start_dim=1）
flatten = nn.Flatten()
# 3. 执行展平
output = flatten(x)

# 4. 查看结果
print("展平后张量形状：", output.shape)  # 输出：torch.Size([2, 784])
# 计算逻辑：1×28×28 = 784，保留batch维度（2），后续维度合并为784
print("展平后每个样本的特征数：", output.shape[1])  # 输出：784