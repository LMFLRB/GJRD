import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from math import sqrt, floor
from torchvision.models import resnet18, resnet34, resnet50, resnet101 #, resnet152

from cytoolz.itertoolz import sliding_window
slide = lambda series: sliding_window(2,series)


# __all__ = ['RESNET18', 
#            'RESNET34', 
#            'RESNET50', 
#            'RESNET101', 
#            'Flatten',
#            'ADDneck',
#            'FeatureResnet',
#            'FeatureConv2d',
#            'FeatureFC',
#            'Predictor',
#            'Attension'
#            ]
# try:
#     RESNET18 = resnet18(weights='ResNet18_Weights.DEFAULT', feature_only=True)
# except:
#     pass
# try:
#     RESNET34 = resnet34(weights='ResNet34_Weights.DEFAULT', feature_only=True)
# except:
#     pass
# try:
#     RESNET50 = resnet50(weights='ResNet50_Weights.DEFAULT', feature_only=True)
# except:
#     pass
# try:
#     RESNET101 = resnet101(weights='ResNet101_Weights.DEFAULT', feature_only=True)
# except:
#     pass
# ResNet152 = resnet152(weights='ResNet152_Weights.DEFAULT', feature_only=True)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class myRNN(nn.Module):
    def __init__(self, network, **kwargs) -> None:
        super().__init__()
        self.model = getattr(nn, network)(**kwargs)
    def forward(self, inputs):
        res=self.model(inputs)
        # self.hidden=res[1]
        return res[0]
    
class myMaxPool(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.model=nn.MaxPool2d(*args, return_indices=True)
    def forward(self, inputs):
        outputs, indices = self.model(inputs)
        self.indices=indices
        return outputs

class myMaxUnPool(nn.Module):
    def __init__(self, *args, output_size=None) -> None:
        super().__init__()
        self.model=nn.MaxUnpool2d(*args)
        self.output_size=output_size
        self.indices=None
    def forward(self, inputs):
        output_size = tuple([inputs.shape[0],inputs.shape[1], self.output_size, self.output_size])
        return self.model(inputs,self.indices,output_size)

class Gate(nn.Module):
    def __init__(self, dimension) -> None:
        super().__init__()
        self.dimension = dimension
        self.model  = nn.Sequential(nn.Linear(dimension, dimension),nn.Sigmoid())
    def forward(self, inputs):
        att_forward  =  inputs[..., :self.dimension]#.flip(1)
        att_backward =  inputs[..., self.dimension:]
        embedding = att_forward*self.model(att_forward) + att_backward*self.model(att_backward)        
        return embedding

class Reshape2D(nn.Module):
    def __init__(self, patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
    def forward(self, inputs):
        """
        Flatten: to concatenate the remaining observations for each timestep, producing one vector for each sequence.
        """
        new_shape = [inputs.shape[0],-1] + self.patch_size 
        return inputs.view(new_shape)
    
class Flatten(nn.Module):
    # def __init__(self, *args, start_dim=None, **kwargs) -> None:
    #     super().__init__(*args, start_dim=1, **kwargs)
    #     self.start_dim = start_dim if start_dim is not None else (args[0] if len(args)>0 else 1)

    def forward(self, input, start_dim=1):
        return input.flatten(start_dim)
    
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

class ADDneck(nn.Module):
    def __init__(self, 
                 hiddens=[2048,1024,256,128], 
                 kernels=[3,3,3],
                 strides=[1,1,1],
                 paddings=[0,1,0],
                 patchsize=224,
                 **kwargs):
        super(ADDneck, self).__init__()
        self.stride = strides[0]
        self.patchsize = patchsize
        models=[]
        for hidden,kernel,stride,padding in zip(slide(hiddens),kernels,strides,paddings):
            self.patchsize = floor( (self.patchsize+2*padding-kernel)/stride +1 )
            models.append(
                nn.Sequential(nn.Conv2d(hidden[0],hidden[1],kernel,stride,padding),
                            nn.BatchNorm2d(hidden[1]),
                            nn.ReLU(inplace=True),
                            )
                )
        self.model=nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)
      
class FeatureResnet(nn.Module):
    def __init__(self,resnet_type:str='ResNet50',):
        super(FeatureResnet, self).__init__()   
        self.network=resnet_type
        try:     
            n_layers = int(resnet_type[6:])
            self.model=globals()[resnet_type.lower()](
                                weights=f'ResNet{n_layers}_Weights.DEFAULT', 
                                feature_only=True)
        except:
            self.model=globals()[resnet_type.lower()](
                                pretrained=True,
                                feature_only=True)

    def forward(self, x, **kwargs):  
        if kwargs.get('transforms') is not None:
            x=kwargs.get('transforms')(x)
        return self.model(x)

class FeatureConv2d(nn.Module):
    def __init__(self,in_channel=3, 
                    hiddens=[64,64,128],
                    kernels=[5,5,5],
                    strides=[1,1,1],
                    paddings=[2,2,2],
                    lambd=1.0,
                    patchsize=32,
                    use_mps=[True,True,False],
                    kernel_mp=3,
                    stride_mp=2,
                    padding_mp=1):
        super(FeatureConv2d, self).__init__()
        
        self.lambd = lambd
        # get the output size
        models=nn.ModuleList([])
        nodes=[in_channel]+hiddens
        self.patchsize = patchsize
        for i, (hidden,kernel,stride,padding,use_mp) in enumerate(zip(slide(nodes),kernels,strides,paddings,use_mps)):
            layers = [nn.Conv2d(hidden[0],hidden[1],kernel,stride,padding),nn.BatchNorm2d(hidden[1]),nn.ReLU()]
            self.patchsize = floor( (self.patchsize+2*padding-kernel)/stride +1 )
            if use_mp:
                self.patchsize = floor( (self.patchsize+2*padding_mp-kernel_mp )/stride_mp +1 )
                layers.append(nn.MaxPool2d(kernel_mp,stride_mp,padding_mp))
            models.append(nn.Sequential(*layers))
        models.append(Flatten())
        self.model=nn.Sequential(*models)

    def forward(self, x,reverse=False):   
        x = self.model(x)
        if reverse:
            x = grad_reverse(x, self.lambd)
        return x

class FeatureFC(nn.Module):
    def __init__(self, 
                 hiddens=[2048,256,256,256],
                 activation='ReLU',
                 use_batchnorm=True,
                 use_dropout=False,
                 **kwargs):
        super(FeatureFC, self).__init__()

        models=[]
        for input,output in slide(hiddens):
            layers=[nn.Linear(input,output)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(output))
            layers.append(getattr(nn, activation)())
            if use_dropout:
                layers.append(nn.Dropout1d(0.3))
            models.append(nn.Sequential(*layers))
        self.model = nn.Sequential(*models)
    
    def forward(self, x):
        return self.model(x)
    
class Attension(nn.Module):
    def __init__(self, feature_dim, attension_dim:int=0):
        super().__init__()
        self.attension_dim = attension_dim
        self.feature_dim = feature_dim        
        self.va=nn.Sequential(nn.Linear(feature_dim, feature_dim),nn.Tanh())
        self.ua=nn.Linear(feature_dim, feature_dim, bias=False)
        self.Lambda=nn.Softmax(dim=self.attension_dim)
        
    def forward(self, in_seqs):
        return (self.Lambda(self.va(in_seqs)*self.ua(in_seqs))*in_seqs).sum(self.attension_dim)

class BilateralAttension(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.forward_attension=Attension(feature_dim)
        self.backward_attension=Attension(feature_dim)
        self.gate  = nn.Sequential(nn.Linear(feature_dim, feature_dim),nn.Sigmoid())
                
    def forward(self, hidden):
        att_forward  =  self.forward_attension(hidden[..., :self.feature_dim].flip(1))
        att_backward =  self.backward_attension(hidden[..., self.feature_dim:] )
        embedding = att_forward*self.gate(att_forward) + att_backward*self.gate(att_backward)
        
        return embedding
  
class Predictor(nn.Module):
    def __init__(self, 
                 n_feat:int=2048,
                 n_class:int=10,
                 cls_hiddens:list=None,
                 prob=0.1):
        super(Predictor, self).__init__()
        self.prob = prob
        if cls_hiddens is None:
            self.model = nn.Sequential(nn.Dropout(prob),
                                        nn.Linear(n_feat, n_class),
                                        nn.BatchNorm1d(n_class),
                                        nn.Softmax(-1))
        else:
            nodes = [n_feat]+cls_hiddens+[n_class]
            self.model = nn.Sequential(*[
                                nn.Sequential(nn.Dropout(prob),
                                              nn.Linear(n_in, n_out),
                                              nn.BatchNorm1d(n_class),
                                              ) for (n_in, n_out) in enumerate(slide(nodes))
            ].extend(nn.Softmax(dim=-1)))

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        return self.model(x)
