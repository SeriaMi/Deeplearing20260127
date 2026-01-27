import  torchvision
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('cifar10')
trans_compose = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='dataset_cifar10',
train=True,transform=trans_compose,download=True)
test_set = torchvision.datasets.CIFAR10(root='dataset_cifar10',
train=False,transform=trans_compose,download=True)
for i in range(10):
    img,target = train_set[i]
    writer.add_image('cifar10_train',img,i)
writer.close()


