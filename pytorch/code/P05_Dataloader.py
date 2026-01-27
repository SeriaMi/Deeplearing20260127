#dataloader
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data = torchvision.datasets.CIFAR10(root='./pytorch/Dataset/dataset_cifar10',
train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,drop_last=False,num_workers=0)

writer = SummaryWriter('./pytorch/logs/cifar10_test_dataloader')
for epoch in range(2):
    step = 0
    for data,target in test_loader:
        writer.add_images(f'cifar10_test_{epoch}',data,step)
        step += 1
writer.close()