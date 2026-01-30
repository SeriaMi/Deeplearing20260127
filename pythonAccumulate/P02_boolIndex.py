import torch
# 布尔索引是 PyTorch 中针对张量的筛选机制：用与张量同形状的布尔张量作为索引时，
# 会只保留布尔值为 True 对应的位置的元素。
#Test
test_tensor = torch.tensor([1,2,3,4,5,6,4,7,5])
test_boolTensor = torch.tensor([True, False, True, False, True, False, True, False, True])
test_output = test_tensor[test_boolTensor]
print(test_output)

