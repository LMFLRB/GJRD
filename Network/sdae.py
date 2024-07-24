from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List, Union, Any
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from math import floor
from copy import deepcopy as copy
from tqdm import tqdm
from .base import *



def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.

    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)

class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        name:str="sdae",
        hidden_dims: list=[],
        input_dim: list=[],
        latent_dim: int=10,
        network: str="MLP",
        activation: str="ReLU",
        final_activation: Optional[str] = "Tanh",
        use_batchnorm: bool=True,
        use_maxpool: bool=True,
        weight_init: bool=True,
        weight_init_func: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
        patch_size:List[int]=[28, 28],
        kernel_sizes:Union[list,int]=3,
        strides:Union[list,int]=2, 
        paddings:Union[list,int]=1, 
        dilations:Union[list,int]=1,
        kernel_mps:Union[list,int]=2,
        **kwargs,
    ):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].

        :param hidden_dims: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.name = name
        self.network = network
        self.use_batchnorm=use_batchnorm
        self.use_maxpool=use_maxpool
        self.input_dim = input_dim
        self.activation = copy(activation)
        self.dimensions = [self.input_dim]+(hidden_dims[network] if isinstance(hidden_dims, dict) else hidden_dims)
        self.bidirectional = kwargs.get('bidirectional')
        self.orig_patch_size=copy(patch_size)

        
        self.kernel_sizes=kernel_sizes
        self.strides= strides
        self.paddings=paddings
        self.dilations=dilations
        self.kernel_mps=kernel_mps
        
        reshape_encode=None
        reshape_decode=None
        if network=="MLP":
            output_padding=[0]
            flatten_dim = self.dimensions[-1]
            self.kwargs = {}
        elif network=="CNN":
            patch_size = patch_size[0]
            conv_channels = [patch_size]
            if use_maxpool:
                conv_channels_mp=[]
            for n in range(len(self.dimensions)-1):
                kernel_size,stride,padding,dilation,kernel_mp=self.get_layerwise_params(n)
                patch_size = floor( (patch_size+2*padding-dilation*(kernel_size-1)-1)/stride +1 ) 
                if use_maxpool:
                    conv_channels_mp.append(patch_size)
                    patch_size = floor( (patch_size-kernel_mp )/stride +1 )
                conv_channels.append(patch_size)                
            reshape_encode="flatten"
            reshape_decode=[copy(patch_size), copy(patch_size)]
            conv_channels.reverse()
            if use_maxpool:
                conv_channels_mp.reverse()
                output_size=[]
            output_padding=[]

            for i in range(len(self.dimensions)-1):
                if self.use_maxpool:
                    patch_size = conv_channels_mp[i]
                    output_size.append(patch_size)

                patch_size_t = (patch_size-1)*stride-2*padding+dilation*(kernel_size-1)+1
                delta = conv_channels[i+1]-patch_size_t if patch_size_t<conv_channels[i+1] else 0
                patch_size = patch_size_t+delta  
                output_padding.append(delta)

            self.kwargs = dict(kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding, 
                               dilation=dilation,
                               output_padding=output_padding)
            if use_maxpool:
                self.kwargs = dict(self.kwargs,
                               kernel_mp=kernel_mp, 
                               output_size=output_size)
            
            flatten_dim = self.dimensions[-1]*conv_channels[0]**2 
        else:# RNN
            self.kwargs = dict(batch_first=kwargs["batch_first"],
                               num_layers=kwargs['num_layers'], 
                               bidirectional=kwargs['bidirectional'], 
                               dropout=kwargs['dropout'])
            self.activation = None
            flatten_dim = self.dimensions[-1]*(2 if kwargs['bidirectional'] else 1 )
            reshape_encode = ""
            reshape_decode = [flatten_dim]

        self.kwargs["direction"]="forward" 
        # construct the encoder 
        encoder_units = self.build_units(self.dimensions, self.activation, network, **self.kwargs)      
        if network in ["MLP", "CNN"]:
            encoder_units.extend( 
                    self.build_units([flatten_dim, latent_dim], kwargs.get('latent_activation'), network, reshape=reshape_encode, 
                                **dict(self.kwargs, use_batchnorm=True)))
        self.encoder = nn.Sequential(*encoder_units)
        
        # construct the decoder
        self.dimensions.reverse()
        self.kwargs["direction"]="backward"
        decoder_units = [] if network not in ["MLP", "CNN"] else \
            self.build_units([latent_dim, flatten_dim], activation, network, reshape=reshape_decode, 
                                    **dict(self.kwargs))     
 
        decoder_units.extend( 
                        self.build_units(self.dimensions, self.activation, network, 
                                    **dict(self.kwargs, final_activation=final_activation))) 
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        if weight_init:
            for layer in concat([self.encoder, self.decoder]):
                weight_init_func(layer.main.weight, layer.main.bias, gain)

    def get_layerwise_params(self, n_layer):
        return tuple([getattr(self, para)[n_layer] if isinstance(getattr(self, para), list) 
                      else getattr(self, para) for para in ["kernel_sizes",
                                                            "strides",
                                                            "paddings",
                                                            "dilations",
                                                            "kernel_mps"]
                    ])

    def build_units(self, dimensions: Iterable[int], 
                    activation: Optional[Union[torch.nn.Module, str]]=None,
                    network:str="MLP",
                    reshape=None,
                    **kwargs,
    ) -> List[torch.nn.Module]:
        """
        Given a list of dimensions and optional activation, return a list of units where each unit is a linear
        layer followed by an activation layer.

        :param dimensions: iterable of dimensions for the chain
        :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
        :return: list of instances of Sequential
        """
        def single_unit_linear(in_dimension: int, out_dimension: int, activation: str, **kwargs) -> torch.nn.Module:
            unit = []
            if reshape=="flatten" or network=="MLP": #flatten when handling image data
                unit.append(("flatten", Flatten()))
            unit.append(("main", nn.Linear(in_dimension, out_dimension)))
            if isinstance(reshape, List) and network!="MLP": #reshape to [bn,channles,patchsize] when handling image data
                if kwargs.get("use_batchnorm") is not None:
                    unit.append(("batchnorm", nn.BatchNorm2d(out_dimension)))
                unit.append(("reshape", Reshape2D(reshape)))
            if activation is not None:
                unit.append(("activation", getattr(nn, activation)()))
            return nn.Sequential(OrderedDict(unit))
        
        def single_unit_conv2d(in_dimension: int, out_dimension: int, activation: str, **kwargs) -> torch.nn.Module:      
            direction = kwargs.pop("direction")
            if kwargs.get("kernel_mp") is not None:
                kernel_mp = kwargs.pop("kernel_mp")
                if kwargs.get("output_size") is not None:
                    output_size = kwargs.pop("output_size")
                    
            unit = []
            if activation is not None:     
                act=activation if isinstance(activation, nn.Module) else getattr(nn, activation)()  

            if direction=="forward":
                unit.append(("main", nn.Conv2d(in_dimension, out_dimension, **kwargs)))
                if activation is not None:
                    unit.append(("activation", act))
                if 'kernel_mp' in locals():
                    unit.append(("max_pool", myMaxPool(kernel_mp, 2, 0)))
            else:
                if 'kernel_mp' in locals():
                    unit.append(("max_unpool", myMaxUnPool(kernel_mp, 2, output_size=output_size)))
                unit.append(("main", nn.ConvTranspose2d(in_dimension, out_dimension, **kwargs)))
                if activation is not None:
                    unit.append(("activation", act))
            
            return nn.Sequential(OrderedDict(unit))
            
        def single_unit_rnn(in_dimension: int, out_dimension: int, activation: str, **kwargs) -> torch.nn.Module: 
            direction=kwargs.pop("direction")
            if direction=="backward" and kwargs["bidirectional"]:
                units=[("gate", Gate(in_dimension))]
            else:
                units=[]
            units.append(("main", myRNN(network, input_size=in_dimension, hidden_size=out_dimension, **kwargs)))        
            return nn.Sequential(OrderedDict(units))

        if network=="MLP":
            unit_func = single_unit_linear
        elif network=="CNN":
            unit_func = single_unit_conv2d if reshape is None else single_unit_linear
        else:
            unit_func = single_unit_rnn  if reshape is None else single_unit_linear

        if kwargs.get("final_activation") is not None:
            final_activation=kwargs.pop("final_activation")
        if network=="CNN":
            output_padding=kwargs.pop("output_padding") 
            if kwargs.get("kernel_mp") is not None:
                output_size=kwargs.pop("output_size") 
        units = []
        for i, (embed_dim, hidden_dim) in enumerate(sliding_window(2, dimensions)):
            single_kwargs=copy(kwargs) 
            if network=="CNN" and kwargs["direction"]=="backward":          
                single_kwargs["output_padding"]=output_padding[i]                  
                if kwargs.get("kernel_mp") is not None:         
                    single_kwargs["output_size"]=output_size[i]  
            if unit_func.__name__=="single_unit_linear":
                single_kwargs['reshape']=reshape
            if i==len(dimensions)-2 and kwargs.get("final_activation") is not None:
                cur_activation=final_activation
            else:
                cur_activation=activation
            units.append(unit_func(embed_dim, hidden_dim, cur_activation, **single_kwargs))

        return units
    
    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.

        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 1) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index], self.decoder[-(index + 1)], 
    
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder(batch)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:        
        self.make_unmaxpool_consistent()
        return self.decoder(embeddings)
    
    def make_unmaxpool_consistent(self):
        if self.use_maxpool:
            l=len(self.encoder)-1
            for num in range(l):
                setattr(self.decoder[l-num].max_unpool, "indices", self.encoder[num].max_pool.indices)
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))

    def predict(self, batch: torch.Tensor, **kwargs) -> tuple:
        embedding = self.encode(batch)
        reconst = self.decode(embedding)
        return (reconst.detach(),
                batch.detach(),
                embedding.detach() 
                )
    
    def loss_function(self, results, loss_caller=None, **kwargs) -> dict:
        """
        Computes the loss.        
        param results:
            [-2]~reconst: reconstructions when using autoencoder
            [-1]~input: the original input
        param kwargs:
            all loss cretirion object used            
        return:
            loss_dict
        """
        origin=(torch.cat([results[1], results[1].flip(1)],-1) 
                if self.network in ["GRU", "RNN", "LSTM"] and self.bidirectional==True \
                else results[1])
        inputs = dict(reconst=results[0].flatten(1), origin=origin.flatten(1))
        if len(results)>2:
            inputs=dict(inputs, embedding=results[2])
        # if self.generative:
        #     inputs = dict(inputs, mu=results[-4], log_var=results[-3])
        
        loss = loss_caller(inputs, **kwargs)
        
        return loss
    
    