import torch, os
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader
from torchvision import transforms, datasets
from typing import List, Any, Union
import numpy as np
import pandas as pd
from copy import deepcopy as copy
from scipy.io import loadmat, savemat
import pickle
from abc import abstractmethod
from .utils import get_father_path

class Transform(transforms.Normalize):
    def __init__(self, name='normalize', mean=0,std=1) -> None:
        super(Transform, self).__init__(mean=mean,std=std)
        self.name = name.lower()
        self.range = None
        self.base = None
        self.device = None

    def getattr(self, __name: str) -> Any:
        return super(Transform, self).__getattribute__(__name)

    def forward(self, dataIn) -> torch.Tensor:        
        range = self.range
        base = self.base
        name = self.name.lower()
        if isinstance(dataIn, torch.Tensor):     
            dataIn = dataIn.cpu().numpy() if dataIn.is_cuda else dataIn.numpy()
        if name == 'sigmoid':
            return torch.from_numpy(1.0/(1+np.exp(-dataIn)))
        if name == 'tanh':
            e = np.exp(dataIn)
            e_= 1/e
            err = (e-e_)/(e+e_)
            return torch.from_numpy(err)
        
        flag = (range, base) if (range is None and base is None) else (range[0], base[0])

        if flag[0] is None and flag[1] is None:
            dataIn = np.vstack(dataIn) if isinstance(dataIn, List) else dataIn
            data = dataIn.reshape(-1, dataIn.shape[-1])
            if name == 'min_max':
                self.base = data.min(0)
                self.range = data.max(0) - self.base
                data_normlized = (dataIn - self.base) / self.range
            elif name == 'standardize':
                self.base = data.mean(0)
                self.range = data.std(0)
                data_normlized = (dataIn - self.base) / self.range
            elif name == 'softmax':
                self.base = data.max(0)
                self.range = 1
                data_normlized = np.exp(dataIn - self.base)
        else:
            if name in ['min_max', 'standardize']:
                data_normlized = (dataIn - self.base) / self.range   
            elif name == 'softmax':
                data_normlized = np.exp(dataIn - self.base)

        return torch.from_numpy(data_normlized)

    def inverse(self, data) -> torch.Tensor:
        range = self.range
        base = self.base
        name = self.name.lower()
        if isinstance(data, torch.Tensor):
            self.device = data.device       
            data = data.cpu().numpy() if data.is_cuda else data.numpy()
        if name == 'sigmoid':
            data_inverse = np.log(data/(1-data))
        elif name == 'tanh':
            data_inverse = 0.5*np.log((1+data)/(1-data))
        elif name == 'softmax':
            data_inverse = base + np.log(data) if base is not None else data
        else:
            data_inverse = data*(range)+base if base is not None else data

        return  torch.from_numpy(data_inverse).to(self.device)

class MyTimeSeries(Dataset):
    def __init__(self, name:str="SpokenArabicDigits", root:str=None, split:str="train", **kwargs) -> None:   
        super(MyTimeSeries, self).__init__()      
        self.root=root
        self.name=name
        self.split=split
        self.data_path = os.path.join(root, name, f"{name}_{split.upper()}")
        if not os.path.exists(self.data_path+".mat"):
            self.prepare_data()
        self.data = self.split_with_given_clusters(torch.load(self.data_path+".mat"), **kwargs)      
        self.dataset, self.labels = self.data["dataset"], self.data["labels"]     
        try:
            self.transform = Transform(kwargs["normalize_type"])
            if not self.transform.name in ["tanh", "sigmoid"]:
                temp=np.vstack([seq for seq in self.dataset])
                temp=self.transform(temp)
                del temp
            if type(self.dataset)==list:
                self.dataset = [self.transform(seq) for seq in self.dataset]
            else:
                self.dataset = self.transform(self.dataset)
        except:
            pass   
        self.clusters=list(set(self.labels))     
        self.n_cluster=len(self.clusters)
        self.n_sample = len(self.dataset)
        self.feature_dim=self.dataset[0].shape[-1]
        self.max_len = 0    
        self.min_len = 100000     
        for seq in self.dataset: 
            if self.max_len < seq.shape[0]: 
                self.max_len = seq.shape[0]
            if self.min_len > seq.shape[0]: 
                self.min_len = seq.shape[0]    
        self.pre_padding = kwargs.get("pre_padding") if kwargs.get("pre_padding") is not None else False
        if self.pre_padding:
            self.padding_value=kwargs.get("padding_value") if kwargs.get("padding_value") is not None else 0
            
    
    def prepare_data(self) -> Any:
        raise NotImplementedError 

    @abstractmethod
    def split_with_given_clusters(self, data: dict={}, **kwargs):
        dataset, labels = data["dataset"], data["labels"]
        clusters=kwargs.get("clusters")
        n_cluster=kwargs.get("n_cluster")
        len_clusters = len(set(list(labels)))
        if clusters is None and n_cluster  is None:
            pass
        else:
            dataset_split, labels_split = [], []
            if clusters is not None:
                if len_clusters<=len(clusters):
                    pass
                else:
                    for label in clusters:
                        indics = np.where(labels==label)[0]
                        labels_split =labels_split +labels[indics].tolist()
                        dataset_split=dataset_split+[dataset[index] for index in indics]
                    data["dataset"], data["labels"] = copy(dataset_split), np.array(copy(labels_split))
                    del dataset_split, labels_split
                return data
            elif n_cluster is not None:
                if len_clusters<=n_cluster:
                    pass
                else:
                    for label in list(set(labels))[:n_cluster]:
                        indics = np.where(labels==label)[0]
                        labels_split =labels_split +labels[indics].tolist()
                        dataset_split=dataset_split+[dataset[index] for index in indics]
                    data["dataset"], data["labels"] = copy(dataset_split), np.array(copy(labels_split))
                    del dataset_split, labels_split
        return data


    @abstractmethod
    def getattr(self, __name: str) -> Any:
        return super(MyTimeSeries, self).__getattribute__(__name)
    
    @abstractmethod
    def __getitem__(self, index):
        if self.pre_padding:
            padding = torch.full((self.max_len -self.dataset[index].shape[0],) + self.dataset[index].shape[1:],
                                self.padding_value, dtype=self.dataset[index].dtype)
            return torch.cat([self.dataset[index], padding], dim=0).to(torch.float32), int(self.labels[index])
        else:
            return self.dataset[index], int(self.labels[index])
        
    @abstractmethod
    def __len__(self):
        return len(self.dataset)

class SpokenArabicDigits(MyTimeSeries):    
      def prepare_data(self):
        """
        Number of Instances: 8800
        Number of Attributes: 13
        Each line on the data base represents 13 MFCCs coefficients in the increasing order separated by spaces. 
        This corresponds to one analysis frame. 
        The 13 Mel Frequency Cepstral Coefficients (MFCCs) are computed with the following conditions;
            Sampling rate: 11025 Hz, 16 bits
            Window applied: hamming
            Filter pre-emphasized: 1-0.97Z^(-1)
        Each line in Train_Arabic_Digit.txt or Test_Arabic_Digit.txt represents 13 MFCCs coefficients in
            the increasing order separated by spaces. This corresponds to one analysis frame.
            Lines are organized into blocks, which are a set of 4-93 lines separated by blank lines and
            corresponds to a single speech utterance of an spoken Arabic digit with 4-93 frames.
            Each spoken digit is a set of consecutive blocks.

        In Train_Arabic_Digit.txt there are 660 blocks for each spoken digit .The first 330 blocks
            represent male speakers and the second 330 blocks represent the female speakers. Blocks 1-660
            represent the spoken digit "0" (10 utterances of /0/ from 66 speakers), blocks 661-1320 represent
            the spoken digit "1" (10 utterances of /1/ from the same 66 speakers 33 males and 33 females), and so on up to digit 9."
                
        """
        n_blocks_each_digit = 660 if self.split=="train" else 220
        n_repeat_each_digit = 10
        import re
        with open(self.data_path+".txt", 'rb') as f:
            lines = f.read().decode('utf-8').split("\n")
            self.dataset = []            
            self.n_sample = 0
            frame = []
            for line in lines:
                if re.match(r'^\s*$', line):
                    if len(frame)>0:
                        self.dataset.append(np.stack(frame))
                        self.n_sample=self.n_sample+1
                    frame=[]
                else:
                    line_data = np.stack([float(value) for value in line.split()])
                    frame.append(line_data)
            frame = None
        labels = torch.tensor([i for i in range(10)]).view(-1,1).repeat(1, n_blocks_each_digit).view(-1).numpy()
        partioners = torch.tensor([i for i in range(int(n_blocks_each_digit/n_repeat_each_digit))]).view(-1,1).repeat(1,
                                    n_repeat_each_digit).view(-1).repeat(10).numpy()
        torch.save(dict(dataset=self.dataset, 
                        labels=labels,
                        partioners=partioners,
                        n_sample=self.n_sample), self.data_path+".mat")
 
class CharacterTrajectories(MyTimeSeries):    
    def prepare_data(self, root):
        """
        The characters here were used for a PhD study on primitive extraction using HMM 
        based models. The data consists of 2858 character samples, contained in the cell 
        array 'mixout'. The struct variable 'consts' contains a field consts.charlabels 
        which provides ennummerated labels for the characters. consts.key provides the 
        key for each label. The data was captured using a WACOM tablet. 3 Dimensions 
        were kept - x, y, and pen tip force. The data has been numerically 
        differentiated and Gaussian smoothed, with a sigma value of 2. Data was captured 
        at 200Hz. The data was normalised with consts.datanorm. Only characters with a 
        single 'PEN-DOWN' segment were considered. Character segmentation was performed 
        using a pen tip force cut-off point. The characters have also been shifted so 
        that their velocity profiles best match the mean of the set.
        """
        data = loadmat(os.path.join(root, "CharacterTrajectories", "mixoutALL_shifted.mat"))
        dataset= [sample[:][:].transpose(1,0) for sample in data["mixout"][0]]
        clusters=data["consts"][0][0][3][0]
        labels = data["consts"][0][0][4][0]
        deltaT = data["consts"][0][0][5][0].item()
        units= data["consts"][0][0][10]
        norm_param= data["consts"][0][0][11]
        N = data["consts"][0][0][12][0].item()
        maxshift= data["consts"][0][0][13][0].item()

        torch.save(dict(dataset=dataset[:1433], 
                        labels=labels[:1433],
                        clusters=clusters,
                        deltaT=deltaT,
                        units=units,
                        norm_param=norm_param,
                        n_sample=1433,
                        maxshift=maxshift), 
                os.path.join(root, "CharacterTrajectories", "CharacterTrajectories_TRAIN.mat"))
        torch.save(dict(dataset=dataset[1433:], 
                        labels=labels[1433:],  
                        clusters=clusters,                      
                        deltaT=deltaT,
                        units=units,
                        norm_param=norm_param,
                        n_sample=N-1433,
                        maxshift=maxshift), 
                os.path.join(root, "CharacterTrajectories", "CharacterTrajectories_TEST.mat"))  
    
class CharacterTrajectories1(MyTimeSeries):
    """
    The characters here were used for a PhD study on primitive extraction using HMM 
    based models. The data consists of 2858 character samples, contained in the cell 
    array 'mixout'. The struct variable 'consts' contains a field consts.charlabels 
    which provides ennummerated labels for the characters. consts.key provides the 
    key for each label. The data was captured using a WACOM tablet. 3 Dimensions 
    were kept - x, y, and pen tip force. The data has been numerically 
    differentiated and Gaussian smoothed, with a sigma value of 2. Data was captured 
    at 200Hz. The data was normalised with consts.datanorm. Only characters with a 
    single 'PEN-DOWN' segment were considered. Character segmentation was performed 
    using a pen tip force cut-off point. The characters have also been shifted so 
    that their velocity profiles best match the mean of the set.
    """
    def prepare_data(self):
        data = pd.read_csv(self.data_path+".ts", delimiter="\t").values[41:]
        dataset = []
        labels  = []
        corrupted = []
        for line in data:
            motion = line[0].split(":")
            try:
                line_data = np.stack([np.stack([float(value) for value in m.split(",")]) for m in motion[:-1]])
                dataset.append(line_data.transpose(1,0))
                labels.append(int(motion[-1]))
            except:
                corrupted.append(motion)
                pass
        torch.save(dict(dataset=dataset, 
                        labels=np.array(labels),
                        corrupted=corrupted), self.data_path+".mat")
    
class ECG5000(MyTimeSeries):    
    def prepare_data(self):
        # 1:
        # from .utils import load_from_arff_to_dataframe as load
        # data = load(self.data_path+".arff")
        # 2:
        # from .utils import load_from_txt_to_dataframe as load
        # data = load(self.data_path+".txt")
        # 3:
        pd_data = pd.read_csv(self.data_path+".ts", delimiter="\t")
        attri={}
        dataset, labels = [], []
        data_start = False
        for line in pd_data.values:
            line = line[0].split()
            if data_start:
                line = line[0].split(":")
                data_line, label= tuple(line)
                labels.append(int(label))
                # dataset.append(np.array([np.expand_dims(np.array([float(value)]), axis=1) for value in data_line.split(',')]))
                dataset.append(np.array([np.array([float(value)]) for value in data_line.split(',')]))
            else:
                if line[0][0]=="@":
                    if line[0][1:]=='data':
                        data_start=True
                    elif line[0][1:]=='classLabel':
                        attri[line[0][1:]]= line[2:]
                    else:
                        attri[line[0][1:]]= line[1]

        labels = np.array(labels)
        dataset = np.array(dataset)
        clusters=set(list(labels))

        torch.save(dict(dataset=dataset, 
                        labels=labels,
                        clusters=clusters,
                        n_sample=len(dataset),
                        **attri),                        
                        self.data_path+".mat")
       
class ECG200(MyTimeSeries):
    def prepare_data(self):
        dataset, labels = [], []
        with open(self.data_path+".txt", "r") as f:            
            for line in f.readlines():
                line = line.split()
                label=int(float(line[0]))
                labels.append(label if label==1 else 0)
                dataset.append(np.array([np.array([float(value)]) for value in line[1:]]))
       
        labels = np.array(labels)
        dataset = np.array(dataset)
        clusters=set(list(labels))

        torch.save(dict(dataset=dataset, 
                        labels=labels,
                        clusters=clusters,
                        n_sample=len(dataset)),                        
                        self.data_path+".mat") 
    
class BasicMotion(MyTimeSeries):
    def prepare_data(self):
        # from .utils import load_from_arff_to_dataframe as load
        # data=[]
        # for dim in range(1,7,1):
        #     path = os.path.join(self.root, self.name, self.name+"sDimension"+str(dim)+f"_{self.split.upper()}")
        #     data.append(load(path))
        # pd_data = load(self.data_path)
        from scipy.io.arff import loadarff as load
        data = pd.DataFrame(load(self.data_path+".arff")[0])
        dataset, str_labels = [], []
        for sample in data.values:            
            dataset.append(np.hstack([np.array([r for r in data_dim.real]).reshape(-1,1) for data_dim in sample[0]]))
            str_labels.append(sample[1].decode())
        
        dataset=np.stack(dataset)
        str_labels=np.stack(str_labels)
        clusters=set(list(str_labels))
        labels=np.zeros(len(str_labels))
        for label, str_label in enumerate(clusters):
            indics = np.where(str_labels==str_label)[0]
            labels[indics]=np.ones(len(indics))*label

        
        torch.save(dict(dataset=dataset, 
                        labels=labels.astype(np.int8),
                        clusters=clusters,
                        n_sample=len(dataset),),                        
                        self.data_path+".mat") 

class DynTexture(MyTimeSeries):
    def prepare_data(self):
        # from .utils import load_from_arff_to_dataframe as load
        # data=[]
        # for dim in range(1,7,1):
        #     path = os.path.join(self.root, self.name, self.name+"sDimension"+str(dim)+f"_{self.split.upper()}")
        #     data.append(load(path))
        # pd_data = load(self.data_path)
        from scipy.io.arff import loadarff as load
        data = pd.DataFrame(load(self.data_path+".arff")[0])
        dataset, str_labels = [], []
        for sample in data.values:            
            dataset.append(np.hstack([np.array([r for r in data_dim.real]).reshape(-1,1) for data_dim in sample[0]]))
            str_labels.append(sample[1].decode())
        
        dataset=np.stack(dataset)
        str_labels=np.stack(str_labels)
        clusters=set(list(str_labels))
        labels=np.zeros(len(str_labels))
        for label, str_label in enumerate(clusters):
            indics = np.where(str_labels==str_label)[0]
            labels[indics]=np.ones(len(indics))*label

        
        torch.save(dict(dataset=dataset, 
                        labels=labels.astype(np.int8),
                        clusters=clusters,
                        n_sample=len(dataset),),                        
                        self.data_path+".mat") 

class cache_dataset_into_cuda(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self._cache = dict()
        self.__dict__.update(dataset.__dict__)
        

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.dataset[index])
            self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
            self._cache[index][1] = torch.tensor(
                    self._cache[index][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.dataset)

class tvsDataset(Dataset):
    def __init__(self, 
                 train:bool=True, 
                 name:str="MNIST",
                 patch_size:Union[list,dict]=None,
                 root_dir:str="G:\Data",
                 **kwargs):        
        self.name = name                   
        img_transform = [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.CenterCrop(148),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))]
        if patch_size is not None:
            resize=patch_size if isinstance(patch_size, list) else patch_size[name]
            img_transform.append(transforms.Resize(resize[-2:]))
        img_transform = transforms.Compose(img_transform)
        if name in ['STL10', 'CelebA', 'SVHN']: 
            data_kwargs=dict(root=root_dir, split="train" if train else "test", transform=img_transform, download=True)
            self.ds = getattr(datasets, name)(**data_kwargs)
        elif name in ['SpokenArabicDigits', 'CharacterTrajectories', 'BasicMotion', 'ECG5000', 'ECG200', 'DynTexture']:
            self.ds = globals()[name](name=name, 
                                    root=root_dir,
                                    split="train" if train else "test", 
                                    pre_padding=kwargs.get("pre_padding"),
                                    padding_value=kwargs.get("padding_value"),
                                    normalize_type="tanh",)
        else:           
            data_kwargs=dict(root=root_dir, 
                             train=True if train else False, 
                             transform=img_transform,
                             download=True)
            self.ds = getattr(datasets, name)(**data_kwargs)

        self.feature_shape=self.ds[0][0].shape
        self.is_image = True if (self.feature_shape[-1]==self.feature_shape[-2] and self.feature_shape[-3] in [1,3]) else False
        self.feature_dim=self.ds[0][0].shape[1] if self.is_image else self.ds[0][0].shape[-1]
        
        if hasattr(self.ds, "classes"):
            self.clusters=self.ds.classes 
            self.n_cluster=len(self.clusters)
        else:
            if hasattr(self.ds, "labels"):
                labels = self.ds.labels
            elif hasattr(self.ds, "train_labels"):
                labels=self.ds.train_labels
            else:
                labels=[]
            if len(labels)>0:
                self.clusters=list(set(labels))  
                self.n_cluster=len(self.clusters)
            else:
                self.clusters=[]
                self.n_cluster=0
        
    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.ds[index]

    def __len__(self) -> int:
        return len(self.ds)

class procDataset(Dataset):
    def __init__(self, 
                train:bool=True, 
                name:str="MNIST",
                root_dir:str="G:\Data",
                feature_type:str="linear", 
                resnet_type:str=None,
                **kwargs) -> None:
        super().__init__()
        self.name = name
        self.split= "train" if train else "test"
        self.file = os.path.join(root_dir, 
                                         f"{name}-processed", 
                                         f'{resnet_type}-{feature_type}',
                                         f"{self.split}")
        if os.path.exists(self.file+".mat"):
            self.data = loadmat(self.file+".mat")
        else:
            with open(self.file+".pkl", "rb") as f:
                self.data = pickle.load(f)
        self.features = self.data['features']
        self.labels = self.data['labels'].squeeze()
        self.dataset= TensorDataset(torch.from_numpy(self.features).to(torch.float32),
                                    torch.from_numpy(self.labels).to(torch.long))  
        self.clusters=list(set(self.labels))     
        self.n_cluster=len(self.clusters)
        self.n_sample = len(self.dataset)
        self.feature_shape=self.dataset[0][0].shape
        if len(self.feature_shape)>2:
            if self.feature_shape[-1]==self.feature_shape[-2] and self.feature_shape[-3] in [1,3,512,2048]:
                self.is_image=True            
                self.feature_dim=self.feature_shape[-3]    
        else:
            self.is_image=False            
            self.feature_dim=np.array(self.feature_shape).prod()


    def __getitem__(self, index: int) -> torch.Tensor:        
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


        
def LoadDataset(params, with_processed_data=True, 
                feature_type:str="linear", resnet_type:str=None):   
    # params['root_dir']=os.path.join(get_father_path(3), "Data") 
    params['feature_type']=feature_type 
    params['resnet_type']=resnet_type
    if with_processed_data: 
        if not os.path.exists(
                    os.path.join(params.root_dir, 
                                f'{params.name}-processed', 
                                f'{resnet_type}-{feature_type}',
                                f'done')
                     ):    
            from Network import ImageProcess
            DataProcess = ImageProcess(**params)
            params_=copy(params)
            params_.cuda=False
            to_process = dict(train=tvsDataset(train=True,**params_), 
                              test=tvsDataset(train=False,**params_))
            DataProcess(to_process)

        traindata = procDataset(train=True, **params)
        testdata  = procDataset(train=False,**params)
    else:
        traindata = tvsDataset(train=True, **params)
        testdata  = tvsDataset(train=False,**params)
        
        
    if params.cuda:
        traindata = cache_dataset_into_cuda(traindata)
        testdata  = cache_dataset_into_cuda(testdata)
    return (traindata, testdata)
