#Transform
from torchvision import transforms
from PIL import Image as IMG
#关注不同的方法输入和输出
img = IMG.open('pytorch\Dataset\cats_and_dogs_filtered\train\cats\cat.1.jpg')
print(img)
#compose
trans_compose = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = trans_compose(img)
print(img_tensor)
print(img_tensor.shape)
#normalize
trans_normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
img_tensor = trans_normalize(img_tensor)
print(img_tensor)
print(img_tensor.shape)
#randomcrop
trans_randomcrop = transforms.RandomCrop(224)
img_tensor = trans_randomcrop(img_tensor)
print(img_tensor)
print(img_tensor.shape)
