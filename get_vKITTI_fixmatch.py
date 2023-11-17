import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_processing import TransformFixMatch


class VKITTI_fixmatch(Dataset):
    def __init__(self, path, resolution=(384, 1280)):  # Initialize with your dataset
        self.path = path
        self.resolution = resolution
        self.files = os.listdir(self.path)
        # print(self.files)
        # print(self.resolution)
        # self.trans = transforms.Compose([transforms.Resize(size=(384, 1280))])
        # self.downscale_image = transforms.Resize(self.resolution)  # To Model resolution
        self.transform_fixmatch = TransformFixMatch()
        # self.transform = CenterCrop(self.resolution)
        # transormations for fixmatch


    def __len__(self):                  # upperbound for sample index
        return len(self.files)

    def __getitem__(self, index):

        image_path = os.path.join(self.path, self.files[index])

        data = np.load(image_path, allow_pickle=True)
        # data = self.transform(data)
        depth, image = data['depth'], data['image']
        # print(np.array(depth))
        ## fixmatch transform

        data = self.transform_fixmatch(data)

        # image, depth = data['image'], data['depth']
        image, weak, strong, depth = data['image'], data['weak'], data['strong'], data['depth']

        # print(image.shape)
        image = np.array((image))
        weak = np.array(weak)
        strong = np.array(strong)
        depth = np.array(depth)

        # print(depth)
        return image, weak, strong, depth




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80