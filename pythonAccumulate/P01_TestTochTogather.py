import torch
#torch.gather(input, dim, index)函数:
    # dim为0表示按行进行采样，也就是outputTensor[i,j] = input_tensor[index[i,j],j] 
    # index中列数必须等于input_tensor的列数，否则会报错
    # dim为1表示按列进行采样，也就是outputTensor[i,j] = input_tensor[i,index[i,j]]
    # index中行数必须等于input_tensor的行数，否则会报错

# # ====================== 3. 执行gather采样 ======================
# src_feature = torch.gather(src, 1, img_indices)
# #dim=1,表示按照列进行采样，src_feature[i,j] = src[i,src_feature[i,j]]
# ====================== 1. 定义基础输入张量 ======================
# 被采样的张量：3行4列，数值简单易记
input_tensor = torch.tensor([
    [10, 20, 30, 40],  # 行0
    [50, 60, 70, 80],  # 行1
    [90, 100, 110, 120] # 行2
])
#Test 第一行取10,20,40,二：50 60 70,三：90 100 120
index_tensor_test_col = torch.tensor([
    [0,1,3,0,1],
    [0,1,2,0,1],
    [0,1,3,0,1],
])
dim_test_col = 1
output_test_col = torch.gather(input_tensor,dim_test_col,index_tensor_test_col)
print(output_test_col)
#Test1: 第一列：50；20；70；120；
# output_0[i, j] = input_tensor[index_0[i, j], j]（列 j 不变，行取 index_0 [i,j]）
index_tensor_test_row = torch.tensor(
    [
        [1,0,1,2],#第一列，1代表第一行，采样的就是第一列第一行，
        [0,1,0,1],
        [0,1,0,1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]
)
dim_test_row = 0
output_test_row = torch.gather(input_tensor,dim_test_row,index_tensor_test_row)
print(output_test_row)
