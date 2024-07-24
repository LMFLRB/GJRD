import torch
import numpy as np
import warnings
from typing import Union
from torch import Tensor

from scipy.linalg import logm
from copy import deepcopy as copy

import random

warnings.filterwarnings("ignore")

EPSILON = 1E-20
EPS = 1.0E-40

def triu(X):
    # Sum of strictly upper triangular part
    return X.triu(diagonal=1).sum()

def atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)
    
def p_dist(x, y):
    # x, y should be with the same flatten(1) dimensional
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm2 = torch.sum(x**2, -1).reshape((-1, 1))
    y_norm2 = torch.sum(y**2, -1).reshape((1, -1))
    dist = x_norm2 + y_norm2 - 2*torch.mm(x, y.t())
    return torch.where(dist<0.0, torch.zeros(1).to(dist.device), dist)

def calculate_gram_mat(*data, sigma=1):
    if len(data) == 1:
        x, y = data[0], data[0]
    elif len(data) == 2:
        x, y = data[0], data[1]
    else:
        print('size of input not match')
        return []
    dist = p_dist(x, y)    
    # dist /= torch.max(dist+EPSILON)
    # dist /= torch.trace(dist)
    return torch.exp(-dist / sigma)

def renyi_entropy(x, sigma, alpha=1.001):
    k = calculate_gram_mat(x, sigma=sigma)
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.linalg.eigh(k)[0])
    try:
        eigv = torch.abs(torch.linalg.eigh(k)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
    entropy = (1/(1-alpha))*torch.log2((eigv**alpha).sum(-1))
    return entropy

def joint_entropy(x, y, s_x, s_y, alpha=1.001):
    x = calculate_gram_mat(x, sigma=s_x)
    y = calculate_gram_mat(y, sigma=s_y)
    k = torch.mul(x, y)
    k = k/torch.trace(k)
    try:
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
        
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy
   
def integ_density_2(set1, set2, sigma=1.0):
    distance = p_dist(set1, set2)
    gram = torch.exp(-distance / sigma)
    return gram.sum()/torch.tensor(distance.shape).prod()

def average_pairwise_div(groups:Union[list,tuple], sigmas=None, div_func=None, **kwargs) -> Tensor: 
    n_group=len(groups)
    pd=[div_func(groups[i],groups[j],sigma=None if sigmas is None else sigmas[i,j], 
                     **kwargs) for i in range(n_group) for j in range(i)]
    return sum(pd)*2.0/(n_group**2-n_group)

def mmd(input1, input2, sigma=1):
    m = input1.shape[0]
    n = input2.shape[0]

    kxx = calculate_gram_mat(input1, input1, sigma=sigma)
    kyy = calculate_gram_mat(input2, input2, sigma=sigma)
    kxy = calculate_gram_mat(input1, input2, sigma=sigma)

    d  = (kxx.sum((0,1)) - kxx.trace())/(m*(m-1))
    d += (kyy.sum((0,1)) - kyy.trace())/(n*(n-1))
    d -=  kxy.sum((0,1))*2/(n*m) + torch.tensor(1e-6, device=input1.device)

    # if d.is_cuda:
    #     d = d.cpu()
    return d

def kld(X:Tensor, Y:Tensor, sigma=None, **kwargs):
    kxy= calculate_gram_mat(X, Y, sigma=sigma)
     
    pxx = atleast_epsilon(calculate_gram_mat(X, X, sigma=sigma).mean(-1))
    pyy = atleast_epsilon(calculate_gram_mat(Y, Y, sigma=sigma).mean(-1))
    pxy = atleast_epsilon(kxy.mean(-1))
    pyx = atleast_epsilon(kxy.mean(-2))

    # return (pxx.log()-pxy.log()).mean()
    return (pxx.log()-pyx.log()).mean() + (pyy.log()-pxy.log()).mean()

def csd(sampleSet1, sampleSet2, weight1, weight2, sigma=1.0):
    n1, n2 = sampleSet1.shape[0], sampleSet2.shape[0]
    sampleSet1, sampleSet2 = sampleSet1.view(n1, -1), sampleSet2.view(n2, -1)
    if not sampleSet1.shape[1] == sampleSet2.shape[1]:
        Warning("dimetions of the two input sets isn't consistent")
    
    p1_square = integ_density_2(sampleSet1, sampleSet1, sigma)
    p2_square = integ_density_2(sampleSet2, sampleSet2, sigma)
    p12_cross = integ_density_2(sampleSet1, sampleSet2, sigma)

    cs = -torch.log(p1_square*weight1**2+2*p12_cross*weight1*weight2+p2_square*weight2**2) \
        + torch.log(p1_square)+torch.log(p2_square)
    return cs

def jrd(X:Tensor, Y:Tensor, alpha:float=2.0, **kwargs)->Tensor:
    """fusion with pdf
    """
    k=torch.tensor(alpha-1,device=X.device)
    if 'weights' in kwargs:
        B = Tensor(kwargs['weights']).to(X.device)       
    else:
        B = torch.tensor([0.5,0.5]).to(X.device) 
    p11 = calculate_gram_mat(X, X).mean(-1) #etismate p1 with sampleset1
    p22 = calculate_gram_mat(Y, Y).mean(-1) #etismate p2 with sampleset2
    p12 = calculate_gram_mat(X, Y).mean(-1) #etismate p1 with sampleset2
    p21 = calculate_gram_mat(Y, X).mean(-1) #etismate p2 with sampleset1
    p_cross =  ((B[0]*p11+B[1]*p12)**k + (B[0]*p21+B[1]*p22)**k)/2
        
    cross = -atleast_epsilon(p_cross.mean()).log()
    power = -(atleast_epsilon((p11**k).mean()).log()*B[0]+
              atleast_epsilon((p22**k).mean()).log()*B[1])
    
    return (cross - power)/k# + torch.tensor(2, device=cross.device).log()

def pjrd(*groups:Union[list,tuple], **kwargs) -> Tensor:
    return average_pairwise_div(groups, div_func=jrd, **kwargs)

def pmmd(*groups:Union[list,tuple],  **kwargs) -> Tensor:    
    return average_pairwise_div(groups, div_func=mmd, **kwargs)

def pkld(*groups:Union[list,tuple],  **kwargs) -> Tensor:    
    return average_pairwise_div(groups, div_func=kld, **kwargs)

def gjrd(sampleSet: list, weights: list, sigma=1.0):
    n_cluste = len(sampleSet)
    # n_sample = [set.shape[0] for set in sampleSet]
    
    p = torch.zeros(n_cluste, n_cluste)
    for i in range(n_cluste):
        for j in range(i+1):
            p[i,j]=p[j,i]=integ_density_2(sampleSet[i], sampleSet[j], sigma)*weights[i]*weights[j]

    gjrd = -torch.log(p.sum()) + torch.log(p.trace())

    return gjrd     

def gjrd_cluster(cluster_assignment, kernel_matrix, order=2):    
    G, A = kernel_matrix, cluster_assignment
    n, m = tuple(cluster_assignment.shape)
    k = order-1
    if k==1:
        AGA = atleast_epsilon(A.T.matmul(G).matmul(A)/n**2, eps=EPSILON)
        cross_entropy = -AGA.mean().log()
        power_entropy = -AGA.diag().log().mean()

    else:
        AkT = atleast_epsilon(A**k, EPS).T
        Gak = atleast_epsilon((G.matmul(A))**k, EPS)        

        cross_entropy = -atleast_epsilon(((G.sum(1)/m)**k).sum()/m, EPS).log()
        power_entropy = -atleast_epsilon(AkT.matmul(Gak).diag(), EPS).log().mean()

    return torch.exp(-(cross_entropy-power_entropy)/k)
    # return torch.exp(-(cross_entropy-power_entropy))

def pcsd_cluster(cluster_assignment, kernel_matrix):
    A, G = cluster_assignment, kernel_matrix
    nom = A.T @ G @ A
    dnom = (nom.diag().view(-1,1) @ nom.diag().view(1,-1)).sqrt()

    m = A.shape[1]
    d = triu(atleast_epsilon(nom) / atleast_epsilon(dnom)) * 2/(m*(m-1))
    
    # return -d.log()
    return d

def pjrd_cluster(cluster_assignment, kernel_matrix, order=2.0):    
    def pair_jrd_2(assign_i, assign_j, G):
        power_i = assign_i.view(1,-1).matmul(kernel_matrix).matmul(assign_i.view(-1,1))
        power_j = assign_j.view(1,-1).matmul(kernel_matrix).matmul(assign_j.view(-1,1))
        cross_ij=assign_i.view(1,-1).matmul(kernel_matrix).matmul(assign_j.view(-1,1))
        # cross = atleast_epsilon(power_i+power_j+2*cross_ij, eps=EPSILON).mean().log()
        cross = atleast_epsilon(power_i+power_j, eps=EPSILON).mean().log()
        power = atleast_epsilon(power_i, eps=EPSILON).mean().log() + atleast_epsilon(power_j, eps=EPSILON).mean().log()

        return cross-power
    
    def pair_jrd_alpha(assign_i, assign_j, G, alpha):
        k=alpha-1
        power_i = torch.diag(assign_i).matmul(kernel_matrix).matmul(torch.diag(assign_i))
        power_j = torch.diag(assign_j).matmul(kernel_matrix).matmul(torch.diag(assign_j))
        cross_ij=torch.diag(assign_i).matmul(kernel_matrix).matmul(torch.diag(assign_j))
        cross_ji=torch.diag(assign_j).matmul(kernel_matrix).matmul(torch.diag(assign_i))

        cross_i = ((power_i+cross_ji).mean(-1))**k
        cross_j = ((power_j+cross_ij).mean(-1))**k
        cross = atleast_epsilon(cross_i+cross_j, eps=EPSILON).mean().log()
        power = (atleast_epsilon((power_i.mean(-1))**k, eps=EPSILON).mean().log() 
                +atleast_epsilon((power_j.mean(-1))**k, eps=EPSILON).mean().log())

        return cross-power

    A, G = cluster_assignment, kernel_matrix
    m = A.shape[1]
    if (order-2.0).__abs__()<1.0e-8:
        div = [pair_jrd_2(cluster_assignment[:,i],cluster_assignment[:,j], G) for i in range(m) for j in range(i+1,m)]
        div = sum(div)/len(div)
    else:
        div = [pair_jrd_alpha(cluster_assignment[:,i],cluster_assignment[:,j], G, order) for i in range(m) for j in range(i+1,m)]
        div = sum(div)/len(div)
    return div/(order-1)

def get_kernelsize(features: torch.Tensor, selected_param: Union[int, float]=0.15, select_type: str='meadian'):
    ### estimating kernelsize with data with the rule-of-thumb
    features = torch.flatten(features, 1).detach()
    # if features.shape[0]>300:
    #     idx = [i for i in range(0, features.shape[0])]
    #     random.shuffle(idx)
    #     features = features[idx[:300],:]
    k_features = p_dist(features, features)
    
    if select_type=='min':
        kernelsize = k_features.sort(-1)[0][:, :int(selected_param)].mean()
    elif select_type=='max':
        kernelsize = k_features.sort(-1)[0][:, int(selected_param):].mean()
    elif select_type=='meadian':
        kernelsize = selected_param*k_features.view(-1).median()
    else:
        kernelsize = torch.tensor(1.0, device=features.device)
    
    if kernelsize<EPSILON:
        kernelsize = torch.tensor(EPSILON, device=features.device)

    return kernelsize

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

