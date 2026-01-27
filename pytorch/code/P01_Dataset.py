from torch.utils.data import Dataset
#数据加载 dataset和dataloader
from PIL import Image
import os
class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.path = os.path.join(root_dir,label_dir)
        self.img_list = os.listdir(self.path)
        self.label = label_dir
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path,self.img_list[index]))
        return img,self.label
    def __len__(self):
        return len(self.img_list)

root_dir = r'D:\GraduateWork\DeepLearning20260122\pytorch\Dataset\cats_and_dogs_filtered\train'
label_dir_cats = 'cats'
label_dir_dogs = 'dogs'
cats_dataset = Mydata(root_dir,label_dir_cats)
dog_dataset = Mydata(root_dir,label_dir_dogs)

dataset_All = cats_dataset+dog_dataset
image,_ = dataset_All[0]
image1,_ = dataset_All[1000]
image.show()
image1.show()