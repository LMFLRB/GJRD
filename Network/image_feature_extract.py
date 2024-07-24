import torch, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch import stack, concat
from os.path import join as join
from scipy.io import savemat, loadmat
from .base import *

class ImageProcess(nn.Module):
    name = 'ImageProcessor'
    patch_size=[224,224]
    def __init__(self,                  
                resnet_type:str='ResNet50',
                cuda:bool=True,
                root_dir:str="",
                name:str="",
                **kwargs
                ):
        super(ImageProcess, self).__init__()
        self.resnet_type=resnet_type.upper()
        self.use_cuda=cuda
        self.root_dir=root_dir
        self.dataset=name

        self.model = FeatureResnet(self.resnet_type)
        for param in self.model.parameters():
            param.requires_grad = False
        if cuda:
            self.model = self.model.cuda()
        self.transform = transforms.Resize(self.patch_size, antialias=True)
        self.flags={}
        self.data_path_proc = join(self.root_dir, f"{self.dataset}-processed")
        os.makedirs(self.data_path_proc, exist_ok=True) 
    
    def forward(self, datasets_to_process:dict):
        if os.path.exists(os.path.join(self.data_path_proc, 'done')):
            for split, _ in datasets_to_process.items():
                self.flags[split] = True
            print(f"{self.dataset} has beed processed")
            return
        else:   
            for split, dataset in datasets_to_process.items():
                self.process(dataset, split)
            if not False in self.flags.values():
                with open(os.path.join(self.data_path_proc, 'done'), 'w') as f:
                    f.write('done')
    
    def process(self, dataset, split="train", silent:bool=True):
        try:
            features, labels = [], []
            dataset_ = DataLoader(dataset,batch_size=256,shuffle=False)
            print(f"preprocessing {split} {self.dataset} with {self.model.network}...")
            for bn,(feature,label) in tqdm(enumerate(dataset_)):   
                features.extend(self.extract(feature.to("cuda" if self.use_cuda else "cpu")).tolist())
                labels.extend(label.tolist()) 
            savemat(os.path.join(self.data_path_proc, f"{split}.mat"), 
                    dict(features=np.array(features),
                         labels=np.array(labels))
                    )
            self.flags[split]=True
            
            if not silent:
                print(f"{self.dataset} pre-processing with {split}-set sucessed")
            return        
        except:
            self.flags[split]=False
            Warning(f"{self.dataset} pre-processing with {split}-set failed")

    def extract(self, x):    
        if len(x.shape)==3:
            x = x.unsqeeze(1)
        if x.shape[1]==1:
            x = x.repeat([1,3,1,1])
        if x.shape[-1]!=224:
            x = self.transform(x)
        return self.model(x)
    
    def load_processed(self, split="train"):
        data = loadmat(os.path.join(self.data_path_proc, f"{split}.mat"))
        data = TensorDataset(torch.from_numpy(data['features']),
                            torch.from_numpy(data['labels']))        
        return data
        
