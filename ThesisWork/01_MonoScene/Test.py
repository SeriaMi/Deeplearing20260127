import torch

# ====================== 1. 定义基础参数 ======================
c = 64  # 通道数
h = 32  # 2D特征图高度
w = 32  # 2D特征图宽度
HWD = 1000  # 3D体素总数（模拟值）
hw_plus_1 = h * w + 1  # src的第二个维度大小：1025

# ====================== 2. 重建前文的关键张量 ======================
# ① 重建src：[c, h×w+1] = [64, 1025]（有效特征+零向量）
x2d = torch.randn(c, h, w)  # 模拟2D特征图
src = x2d.view(c, -1)  # 展平为[64, 1024]
zeros_vec = torch.zeros(c, 1).type_as(src)  # 零向量[64,1]
src = torch.cat([src, zeros_vec], 1)  # 拼接后[64, 1025]

# ② 重建img_indices：[c, HWD] = [64, 1000]（含视场内/外索引）
projected_pix = torch.randint(low=0, high=40, size=(HWD, 2))  # 模拟投影坐标
fov_mask = (projected_pix[:, 0] < w) & (projected_pix[:, 1] < h)  # 视场内掩码
pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
img_indices = pix_y * w + pix_x  # 转换为一维索引[1000,]
img_indices[~fov_mask] = h * w  # 视场外索引设为1024
img_indices = img_indices.expand(c, -1).long()  # 扩展为[64, 1000]

# ====================== 3. 执行gather采样 ======================
src_feature = torch.gather(src, 1, img_indices)
#dim=1,表示按照列进行采样，src_feature[i,j] = src[i,src_feature[i,j]]

# ====================== 4. 验证结果 ======================
print(f"src的形状: {src.shape}")  # [64, 1025]
print(f"img_indices的形状: {img_indices.shape}")  # [64, 1000]
print(f"src_feature的形状: {src_feature.shape}")  # [64, 1000]

# 验证视场外体素的特征值是否为0（对应零向量）
out_fov_mask = ~fov_mask.expand(c, -1)  # 扩展到通道维度[64, 1000]
out_fov_features = src_feature[out_fov_mask]
print(f"\n视场外体素的特征值是否全为0: {torch.all(out_fov_features == 0)}")

# 验证视场内体素的特征值与src一致
in_fov_mask = fov_mask.expand(c, -1)
# 取第一个视场内体素的索引和特征
first_in_fov_idx = torch.nonzero(in_fov_mask)[0]  # [通道i, 体素j]
i, j = first_in_fov_idx[0], first_in_fov_idx[1]
src_val = src[i, img_indices[i, j]]
src_feature_val = src_feature[i, j]
print(f"\n视场内体素验证：src[{i}, {img_indices[i,j]}] = {src_val:.4f}")
print(f"src_feature[{i}, {j}] = {src_feature_val:.4f}")
print(f"两者是否相等: {torch.isclose(src_val, src_feature_val)}")