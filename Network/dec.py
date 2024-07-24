import torch, sys, numpy
import torch.nn as nn
from torch.nn import Parameter
from typing import Tuple, Callable, Optional, Union, Any
from copy import deepcopy as copy

from sklearn.cluster import KMeans

from .sdae import StackedDenoisingAutoEncoder as AutoEncoder

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        n_cluster: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param n_cluster: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.n_cluster = n_cluster
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_cluster, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class DeepEmbeddingCluster(nn.Module):
    def __init__(
        self,
        name: str="DEC",
        n_cluster: int=10,
        alpha: float = 1.0,
        cluster_only: bool=True,
        autoencoder: Union[torch.nn.Module, dict]={},
        **kwargs
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param n_cluster: number of clusters
        :param n_latent: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DeepEmbeddingCluster, self).__init__()
        self.name=name
        self.n_cluster = n_cluster
        self.alpha = alpha
        self.cluster_only=cluster_only
        self.n_latent = autoencoder["latent_dim"]
        self.network = autoencoder["network"]
        self.generative = False if autoencoder.get("generative") is None else autoencoder["generative"]
        self.encoder = AutoEncoder(**autoencoder) if isinstance(autoencoder, dict) else autoencoder
        self.kmeans = KMeans(n_clusters=n_cluster, n_init=20)   

        self.assignment = ClusterAssignment(
            self.n_cluster, self.n_latent, self.alpha
        )

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(batch)
    
    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(batch)    
    
    
    def forward(self, batch: torch.Tensor, encode_only:bool=False) -> tuple:
        embedding = self.encode(batch)
        return (self.assignment(embedding), 
                embedding,
                batch) if encode_only else \
                (self.assignment(embedding), 
                embedding, 
                self.decode(embedding), 
                batch)

    def predict(self, batch: torch.Tensor, encode_only:bool=False) -> tuple:
        return tuple(output.detach() for output in self.forward(batch, encode_only))
        
    def loss_function(self, 
                      results: Union[tuple, torch.Tensor], 
                      loss_caller: Any, 
                      **kwargs) -> dict:
        if self.cluster_only:
            inputs=dict(assign=results[0])
        else:
            inputs=dict(assign=results[0], embedding=results[1], reconst=results[2].flatten(1), origin=results[3].flatten(1)) 
        return loss_caller(inputs, **kwargs)    
    
    def metric_function(self, 
                        assign: Union[torch.Tensor, numpy.ndarray], 
                        labels: Union[torch.Tensor, numpy.ndarray],
                        metric_caller: Any) -> dict:
        if isinstance(assign, torch.Tensor):
            if len(assign.shape)>1 and assign.shape[1]>1:
                preds=assign.max(1)[1].detach().cpu().numpy()  
            else:
                preds=assign.squeeze().detach().cpu().numpy()
        else:
            preds=assign      
        truth=labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
        return metric_caller(truth, preds)
    
    def init_cluster_centers(self, data_loader, cuda: bool=True):
        from tqdm import tqdm
        features, labels = [], []    
        for feature, label in tqdm(data_loader, 
                                   dynamic_ncols=True, 
                                   desc="init_cluster_centers loop",
                                   ):
        # for feature, label in data_iterator:
            if cuda and not feature.is_cuda:
                feature = feature.cuda(non_blocking=True)
            features.append(self.encode(feature).detach().cpu())
            labels.append(label)
        features = torch.cat(features)
        labels = torch.cat(labels).long()
        predicted = self.kmeans.fit_predict(features.numpy())
        cluster_centers = torch.tensor(
            self.kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
        )
        if cuda:
            cluster_centers = cluster_centers.cuda(non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            self.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)        
        
        return torch.from_numpy(predicted).to(labels.device), labels.detach()
    
    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    