# import os
# import numpy as np
# new_filename="/home/gxzhou/face_dataset/AGFW/age_AGFWandUTKFace_tuple.txt"
# sex=["female","male"]
# # dir_file="/home/gxzhou/AGFW_V2/128"
# dir_file="/home/gxzhou/face_dataset/AGFW/cropped/128"
# age=os.listdir(os.path.join(dir_file,sex[0]))
# age=sorted(age)
# age_range=["age_7_14","age_15_24","age_25_34","age_35_44","age_44_54","age_55_94"]
# fileName="/home/gxzhou/face_dataset/UTKFace"
#
# UTK=["7_1_","7_0_","8_1_","8_0_","9_1_","9_0_"]
# with open(new_filename, 'w') as file_to_read:
#     for i in range(len(sex)):
#         for j in range(len(age)):
#             print(int(np.floor((j+1)/2)+1),j)
#             for file in os.listdir(os.path.join(dir_file,sex[i],age[j])):
#                 if j ==0:
#                     print(os.path.join(dir_file,sex[i],age[j],file)+'          '+sex[i]+'       '+age_range[0]+'\n')
#                     file_to_read.write(os.path.join(dir_file,sex[i],age[j],file)+'          '+sex[i]+'       '+age_range[0]+'\n')
#                 else :
#                     print(os.path.join(dir_file,sex[i],age[j],file)+'          '+sex[i]+'       '+age_range[int(np.floor((j-1)/2)+1)]+'\n')
#                     file_to_read.write(os.path.join(dir_file,sex[i],age[j],file)+'          '+sex[i]+'       '+age_range[int(np.floor((j-1)/2)+1)]+'\n')
#
#     for i in range(len(UTK)):
#         if (i+1)%2==0:
#             for file in os.listdir("/home/gxzhou/face_dataset/UTKFace"):
#                 if UTK[i]==file[:4]:
#                     print(os.path.join(fileName,  file) + '          ' + sex[1] + '       ' + age_range[0] + '\n')
#                     file_to_read.write(os.path.join(fileName,  file) + '          ' + sex[1] + '       ' + age_range[0] + '\n')
#         else:
#             for file in os.listdir("/home/gxzhou/face_dataset/UTKFace"):
#                 if UTK[i]==file[:4]:
#                     print(os.path.join(fileName,  file) + '          ' + sex[0] + '       ' + age_range[0] + '\n')
#                     file_to_read.write(os.path.join(fileName,  file) + '          ' + sex[0] + '       ' + age_range[0] + '\n')
import os
import random
import torch
import numpy as np
all_attr_names={"age_7_14":0,"age_15_24":1,"age_25_34":2,"age_35_44":3,"age_44_54":4,"age_55_94":5}
path_v="/home/gxzhou/face_dataset/AGFW/age_AGFWandUTKFace_tuple.txt"
age_range=["age_7_14","age_15_24","age_25_34","age_35_44","age_44_54","age_55_94"]
from torch.utils import data
from torchvision import transforms as T

from PIL import Image
class AGFWDataset(data.Dataset):
    def __init__(self, transform,mode):


        self.transform = transform
        self.mode=mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
    def preprocess(self):



        lines_list= [line.rstrip() for line in open(os.path.join(path_v), 'r')]
        random.seed(1234)
        random.shuffle(lines_list)
        for i, line in enumerate(lines_list):
            split = line.split()


            age=split[2]
            sex = split[1]
            filename= split[0]


            label=all_attr_names[age]
            if sex=="male":
                sex_label=1
            elif sex=="female":
                sex_label=0
            else:
                Exception("sex isn't male or female")

            if (i+1) < 200:
                self.test_dataset.append([filename,label,sex_label])
            else:
                self.train_dataset.append([filename,label, sex_label])


        for value ,name in all_attr_names.items():
            print('{}: {}.'.format(value, name))


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename,label, sex= dataset[index]
        image = Image.open(filename)

        return self.transform(image), torch.tensor(label), torch.tensor(sex)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



def get_loader(  image_size=128,
               batch_size=16, dataset='adience', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(image_size))
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'AGFW':
        dataset = AGFWDataset( transform, mode)
    else:
        Exception("data isn't")


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader