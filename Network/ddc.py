import torch.nn as nn
from torch.optim import Optimizer as Optimizer
from typing import Tuple, Callable, Optional, Union, Any
from torch import Tensor, cat
from numpy import ndarray
from collections import OrderedDict

from Network.base import *
from Network.sdae import StackedDenoisingAutoEncoder as SDAE

      
class DeepDivergenceCluster(nn.Module): 
    def __init__(self, 
                 name: str="DDC_resnet",
                 n_cluster: int=10,
                 encode_only: bool=False,
                 autoencoder: Union[nn.Module, dict]={},
                 use_processed_data: bool=True,
                 resnet_type: str= "resnet18",
                 feature_type:str="conv2d",        
                 **kwargs):
        """
        DDC clustering module
        args:
            n_cluster: the number if clusters to learn/ classes in data to handle
            autoencoder: an autoencoder of parameters dict for constructing an autoencoder
        keargs: paramweters for autoencoder, not limited to the following
            network: mlp or cnn
            framework: autoencoder or only encoder
            n_latent: dimmension of the latent to learn
            generative: whether to learn a generative VAE for autoencoder
            activation: activating function of the middle layer
            final_activation: activating function of the output layer
        """
        super().__init__()        
        self.name = name
        self.n_cluster = n_cluster        
        self.encode_only = encode_only
        self.use_processed_data = use_processed_data
        self.resnet_type = resnet_type
        self.feature_type = feature_type


        # autoencoder.update(self.autoencoder)

        n_layers = int(resnet_type[6:])
        autoencoder.update(dict(input_dim=512 if n_layers in [18,34] else 2048, patch_size=[7,7]))
        if not use_processed_data:
            self.extractor=globals()[resnet_type.lower()](
                                    weights=f'ResNet{n_layers}_Weights.DEFAULT', 
                                    feature_only=True)
            if feature_type=="linear":
                self.extractor = nn.Sequential(self.extractor,
                                       nn.AdaptiveAvgPool2d((1,1)),
                                       Flatten())
            for param in self.extractor.parameters():
                param.requires_grad = False

        if feature_type=="linear":
            autoencoder.update( dict(network="MLP",
                                     hidden_dims=[500,500,1000],
                                    #  hidden_dims=[1024,512,512],
                                    #  hidden_dims=[512,512,1024],
                                     latent_dim = 20,
                                     activation= "ReLU",
                                     use_maxpool=False,)
                               )
        else:
            autoencoder.update( dict(network="CNN",
                                     hidden_dims=[512,256,128],
                                     kernel_sizes=[5,5,5],
                                     strides =[1,1,1],
                                     paddings =[1,1,1],
                                     use_maxpool=False,)
                               )
        
        
        self.input_dim = autoencoder["input_dim"]
        self.n_latent  = autoencoder["latent_dim"]
        self.activation = "ReLU" if kwargs.get("activation") is None else kwargs["activation"]
        self.generative = False if autoencoder.get("generative") is None else autoencoder["generative"]
        self.device=None

        self.encoder = SDAE(**autoencoder) if isinstance(autoencoder, dict) else autoencoder
        
        
        self.assignment  = nn.Sequential(
                                    nn.Linear(self.n_latent, self.n_cluster), 
                                    nn.BatchNorm1d(num_features=self.n_cluster),
                                    getattr(nn, self.activation)(),
                                    nn.Softmax(dim=1))
    
    def pre_extract(self, x)-> Tensor:
        return self.extractor(x)
    
    def encode(self, x) -> Tensor:
        self.device=x.device
        return self.encoder.encode(x)
    
    def decode(self, embedding: Tensor) -> Tensor:
        return self.encoder.decode(embedding)

    def forward(self, input: Tensor) -> tuple:
        pseudo_input = self.pre_extract(input) if hasattr(self, "extractor") else input
        embedding = self.encode(pseudo_input)        
        return  (self.assignment(embedding),
                embedding,
                input
                ) if self.encode_only else \
                (self.assignment(embedding),
                embedding,
                self.decode(embedding),
                pseudo_input
                )
    
    def predict(self, inputs: tuple) -> tuple:
        return tuple(output.detach() for output in self.forward(inputs))
    
    def loss_function(self, 
                      results: Union[tuple, Tensor], 
                      loss_caller: Any, 
                      **kwargs) -> dict:
        inputs=dict(assign=results[0], embedding=results[1])
        if not kwargs.get('cluster_only')==True and len(results)>3:
            inputs=dict(inputs, reconst=results[2].flatten(1), origin=results[3].flatten(1)) 
        
        return loss_caller(inputs, **kwargs)    
    
    def metric_function(self, 
                        assign: Union[Tensor, ndarray], 
                        labels: Union[Tensor, ndarray],
                        metric_caller: Any) -> dict:
        if isinstance(assign, Tensor):
            if len(assign.shape)>1 and assign.shape[1]>1:
                preds=assign.max(1)[1].detach().cpu().numpy()  
            else:
                preds=assign.squeeze().detach().cpu().numpy()
        else:
            preds=assign      
        truth=labels.cpu().numpy() if isinstance(labels, Tensor) else labels
            
        return metric_caller(truth, preds)
    