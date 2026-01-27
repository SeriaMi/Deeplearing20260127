import torch

class Test:
    def __init__(self):
        self.x = torch.rand(2,3)
        self.y = torch.rand(2,3)
    def __call__(self):
        print(f"我是Test类")


test = Test()
test()
