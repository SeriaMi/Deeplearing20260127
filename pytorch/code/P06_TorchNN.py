from torch import nn
import torch
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x+1

net = MyModule()
x = torch.randn(10)
print(net(x))
print(x)