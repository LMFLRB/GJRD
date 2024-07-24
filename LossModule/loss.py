'''
Author: Mingfei Lu
Description: define the loss functions
Date: 2021-10-22 13:13:53
LastEditTime: 2022-11-02 10:37:07
'''

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy as copy
from scipy.linalg import logm
from .utils import *


class CrossEntropy(torch.nn.Module):
    def __init__(self, name='cross_entropy', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = self.criterion(args[0], args[1].type(torch.long))
        return loss
    
class myMSE(torch.nn.Module):
    def __init__(self, name='mse', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.reduction='mean' if kwargs.get('reduction') is None else kwargs.get('reduction')    
        if self.reduction in ['max', 'mean', 'none']:
            self.criterion = torch.nn.MSELoss(reduction=self.reduction)
    def forward(self, *args, **kwargs) -> torch.Tensor:
        reduction = self.reduction
        if len(args)==1 and isinstance(args[0], dict):
            args = list(args[0].values())
        if reduction in ['max', 'mean', 'none']:
            loss = self.criterion(args[0], args[1])
        elif isinstance(reduction, int):
            loss = ((args[0]-args[1])**2)
            for dim in range(len(args[0].shape)-1, reduction, -1):
                loss = loss.mean(dim)
        return loss
      
class KLDLoss(torch.nn.Module):
    def __init__(self, name='kld_loss', **kwargs) -> None:
        super().__init__()
        self.name = name.upper()
        self.criter=torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, *args) -> torch.Tensor:
        if len(args)>1:
            mu, log_var = torch.flatten(args[0],1), torch.flatten(args[1],1)
            loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)  
        else:
            # features=target_distribution(args[0])
            features=F.log_softmax(args[0],1)
            samples= F.softmax(torch.randn_like(features), dim=1)
            # loss = torch.nn.functional.kl_div(features, samples)   
            loss = self.criter(features, samples)
        return loss
     
class CSDLoss(torch.nn.Module):
    def __init__(self, name='CSD', **kwargs) -> None:
        super(CSDLoss, self).__init__()    
        self.name = name.upper()
        self.kernelsize = kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize')
    def forward(self, *args) -> torch.Tensor:
        loss = csd(args[0], args[1], self.kernelsize)
        return loss

############### the remainder for GCSD and GJRD
class PCSD(torch.nn.Module):
    def forward(self, *args):
        return pcsd_cluster(args[0], args[1])

class DDC2Loss(torch.nn.Module):
    def forward(self, A, type='mean', statistic_num=100):
        n = A.shape[0]
        M = A @ A.T
        if type=='mean':
            triu_mean = triu(M) * 2/(n*(n-1))
            return  triu_mean
            # return  triu_mean + (1.0-M.trace()/n)

        elif type=='max':
            idx = np.triu_indices(n, 1)
            return torch.sort(M[idx], descending=True)[0][:statistic_num].mean()
      
class DDC3Loss(torch.nn.Module):
    def forward(self, A, G, type='trace', extra:str='gcsd'):   
        if type=='simplex':     
            eye = torch.eye(A.shape[1]).type_as(A)
            M = calculate_gram_mat(A, eye)
            # criterion = PCSD()
            if extra=='ddc':
                return pcsd_cluster(M, G) # pairwise csd
            elif extra=='gjrd':
                return gjrd_cluster(M, G) # gjrd
        
        elif type=='sparse':
            return A.max(1)[0].mean()
        else:
            return (1.0-(A @ A.T).trace()/A.shape[0])

class GJRD(torch.nn.Module):
    def __init__(self, name='GJRD', **kwargs) -> None:
        super(GJRD, self).__init__()        
        self.name = name.upper()
        self.entropy_order = 2 if kwargs.get('entropy_order') is None else kwargs.get('entropy_order')
    def forward(self, *args) -> torch.Tensor:  
        return gjrd_cluster(args[0], args[1], self.entropy_order)
  
class PJRD(torch.nn.Module):
    def __init__(self, name='PJRD', **kwargs) -> None:
        super(PJRD, self).__init__()        
        self.name = name.upper()
        self.entropy_order = 2 if kwargs.get('entropy_order') is None else kwargs.get('entropy_order')
    def forward(self, *args) -> torch.Tensor:  
        return pjrd_cluster(args[0], args[1], self.entropy_order)

class DECLoss(torch.nn.Module):
    def __init__(self, name='DEC', **kwargs) -> None: 
        super(DECLoss, self).__init__()           
        self.name = name
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        # self.dec = torch.nn.KLDivLoss(reduction="batchmean")
        self.dec = torch.nn.KLDivLoss(size_average=False)
        if self.generative and self.weights.get('regular') is not None:
            self.regular = KLDLoss(reduction="batchmean")
        if self.weights.get('reconst') is not None:            
            self.reconst = myMSE()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("assign") is not None:
            target = target_distribution(args["assign"]).detach()
            loss = dict(loss, dec=self.dec(args["assign"].log(), target)/target.shape[0])
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))
                # loss['loss'] = loss['loss'] + self.weights["reconst"]*reconst  if loss.get('loss') is not None else self.weights["reconst"]*reconst
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass

        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss

class VAELoss(torch.nn.Module):
    def __init__(self, name='VAE', **kwargs) -> None: 
        super(VAELoss, self).__init__()           
        self.name = name.upper()
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        self.reconst = myMSE()
        if (self.generative or self.weights.get('regular') is not None):
            self.regular = KLDLoss()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("hidden") is None and args.get("embedding") is not None:
            args["hidden"] = args["embedding"]
        if args.get("embedding") is None and args.get("hidden") is not None:
            args["embedding"] = args["hidden"]
        
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))                
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass
        # if kwargs['epoch']!=self.epoch_last:
        #     print(f"{loss}\n")
        #     self.epoch_last = kwargs['epoch']
        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss

class DDCLoss(torch.nn.Module):
    def __init__(self, name='DDC', **kwargs) -> None: 
        super(DDCLoss, self).__init__()           
        self.name = name.upper()
        self.kernelsize = torch.tensor(kwargs.get('kernelsize')[0] if isinstance(kwargs.get('kernelsize'), list) else kwargs.get('kernelsize'), device="cuda:0")
        self.kernelsize_adapt = kwargs["kernelsize_adapt"]
        self.kernelparams = kwargs["kernelsize_search_params"]
        self.epoch_last=int(-1)
        self.weights = copy(kwargs.get('weights'))
        self.generative = False if kwargs.get('generative') is None else kwargs["generative"]
        
        if  name=='DDC' and (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = PCSD()
        if (self.weights.get('ddc2') is not None) and self.weights['ddc2']>1.0e-10:
            self.ddc2 = DDC2Loss()        
        if (self.weights.get('ddc3') is not None) and self.weights['ddc3']>1.0e-10:
            self.ddc3 = DDC3Loss()
        if (self.weights.get('reconst') is not None) and self.weights['reconst']>1.0e-10:            
            self.reconst = myMSE()
        if (self.generative and self.weights.get('regular') is not None) and self.weights['regular']>1.0e-10:
            self.regular = KLDLoss()

    def forward(self, args, **kwargs) -> torch.Tensor:
        loss={}
        if args.get("hidden") is None and args.get("embedding") is not None:
            args["hidden"] = args["embedding"]
        if args.get("assign") is not None and args.get("hidden") is not None:
            if self.kernelsize_adapt and kwargs['epoch']!=self.epoch_last and kwargs['is_training']:
            # if self.kernelsize_adapt and kwargs['is_training']:
                from .utils import get_kernelsize
                self.kernelsize = get_kernelsize(args["hidden"], self.kernelparams.param, self.kernelparams.func)
                self.epoch_last = kwargs['epoch']
                  
            A = args["assign"]
            G = calculate_gram_mat(args["hidden"], sigma=self.kernelsize)   
            try:
                csda = self.ddc1(A, G)
                loss = dict(loss,  kernel=self.kernelsize,  ddc1=csda)
                # loss['loss'] = self.weights["ddc1"]*csda
            except:
                pass

            try:
                # eye  = self.ddc2(A, type='mean', enable=kwargs.get('enable'))
                eye  = self.ddc2(A)
                loss = dict(loss, ddc2=eye) 
                # loss['loss'] = loss['loss'] + self.weights["ddc2"]*eye
            except:
                pass
                        
            try:
                # csdm = self.ddc3(A, G, type='simplex', extra=self.name.lower()) # sparse
                # csdm = self.ddc3(A, G, type='sparse') # sparse
                csdm = self.ddc3(A, G,) # trace
                loss = dict(loss, ddc3=csdm)
                # loss['loss'] = loss['loss'] + self.weights["ddc3"]*csdm
            except:
                pass
        try:
            if args.get("origin") is not None and args.get("reconst") is not None:
                if kwargs.get("bidirectional"):
                    origin = torch.flatten(args["origin"],2)
                    reconst = torch.flatten(args["reconst"],2)
                    reconst = self.reconst(torch.cat([origin, origin.flip(-1)],-1), reconst)
                else:
                    reconst = self.reconst(torch.flatten(args["origin"],1), torch.flatten(args["reconst"],1))
                # loss['loss'] = loss['loss'] + self.weights["reconst"]*reconst  if loss.get('loss') is not None else self.weights["reconst"]*reconst
                loss = dict(loss, reconst=reconst)
        except:
            pass
        
        try:
            if args.get("mu") is not None and args.get("log_var") is not None:
                regular = self.regular(args["mu"], args["log_var"])
            else:
                regular = self.regular(args["embedding"])
            # loss['loss'] = loss['loss'] + self.weights["regular"]*regular if loss.get('loss') is not None else self.weights["regular"]*regular
            loss = dict(loss, regular=regular)
        except:
            pass
        # if kwargs['epoch']!=self.epoch_last:
        #     print(f"{loss}\n")
        #     self.epoch_last = kwargs['epoch']
        loss['loss'] = sum([value*self.weights[name] for name, value in loss.items() if name in self.weights])
        loss={key: value if (key=="loss" and kwargs['is_training']) else value.item() for key, value in loss.items()}
        return loss
    
class GJRDLoss(DDCLoss):
    def __init__(self, name='GJRD', **kwargs) -> None:
        super(GJRDLoss, self).__init__(name='GJRD', **kwargs)
        self.name = name.upper()
        if (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = GJRD(**kwargs)

class PJRDLoss(DDCLoss):
    def __init__(self, name='PJRD', **kwargs) -> None:
        super(PJRDLoss, self).__init__(name='PJRD', **kwargs)
        self.name = name.upper()
        if (self.weights.get('ddc1') is not None) and self.weights['ddc1']>1.0e-10:
            self.ddc1 = PJRD(**kwargs)
