import os
import random
import torch
all_attr_names={"(4,":0,"(25,":1,"(0,":2,"(8,":3,"(15,":4,"(38,":5,"(48,":6,"(60,":7}
transport_v={"45":"(48,","35":"(38,","13":"(8,","34":"(25,","36":"(38,","55":"(48,","29":"(25,"}
attr_path_list=["fold_frontal_4_data.txt","fold_frontal_3_data.txt","fold_frontal_2_data.txt","fold_frontal_1_data.txt","fold_frontal_0_data.txt"]
path_v="/home/smhe/Adience"
from torch.utils import data
from torchvision import transforms as T

from PIL import Image
class AGEDataset(data.Dataset):
    def __init__(self,image_dir, transform,mode):
        self.image_dir = image_dir

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
        lines_list=[]
        ou_dict={}

        for attr_path in attr_path_list:

            lines_list=lines_list+ [line.rstrip() for line in open(os.path.join(path_v,attr_path), 'r')][1:]
        random.seed(1234)
        random.shuffle(lines_list)
        for i, line in enumerate(lines_list):
            split = line.split()
            label=[False,False,False,False,False,False,False,False]

            age=split[3]
            face_id = split[2]
            user_id	 = split[0]
            original_image	 = split[1]
            if age not in all_attr_names:
                if age not in ou_dict:
                    continue
                else:
                    age=transport_v[age]
                    label[all_attr_names[age]]=True

            else:
                label[all_attr_names[age]]=True

            if (i+1) < 200:
                self.test_dataset.append([user_id, original_image,label,face_id])
            else:
                self.train_dataset.append([user_id,original_image, label,face_id])


        for value ,name in all_attr_names.items():
            print('{}: {}.'.format(value, name))


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        user_id,original_image, label ,face_id= dataset[index]
        image = Image.open(os.path.join(self.image_dir, user_id,"landmark_aligned_face.{}.{}".format(face_id,original_image)))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



def get_loader(image_dir, crop_size=178, image_size=128,
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

    if dataset == 'adience':
        dataset = AGEDataset(image_dir, transform, mode)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader