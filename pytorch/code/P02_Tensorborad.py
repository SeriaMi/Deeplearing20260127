# TensorBoard 可视化
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter('logs')
image_path_cat = r'pytorch\Dataset\cats_and_dogs_filtered\train\cats\cat.0.jpg'
image_path_dog = r'pytorch\Dataset\cats_and_dogs_filtered\train\dogs\dog.0.jpg'
image_array_cat = np.array(Image.open(image_path_cat))
image_array_dog = np.array(Image.open(image_path_dog))
writer.add_image('animal',image_array_cat,0,dataformats='HWC')
writer.add_image('animal',image_array_dog,1,dataformats='HWC')
x = [i for i in range(100)]
y = [i*2 for i in range(100)]
for i in range(len(x)):
    writer.add_scalar('y=2x',y[i],x[i])
writer.close()