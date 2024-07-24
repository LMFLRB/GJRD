import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
from collections import OrderedDict
from copy import deepcopy as copy


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,  
        decoder: nn.Module,
        corruption: Optional[Union[torch.nn.Module, float]] = None,
        network: str="MLP",
        gain: float = nn.init.calculate_gain("relu"),
    ) -> None:
        """
        Autoencoder composed of two Linear units with optional encoder activation and corruption.

        # :param embedding_dimension: embedding dimension, input to the encoder
        # :param hidden_dimension: hidden dimension, output of the encoder
        :param activation: optional activation unit, defaults to nn.ReLU()
        :param gain: gain for use in weight initialisation
        :param corruption: optional unit to apply to corrupt input during training, defaults to None
        :param tied: whether the autoencoder weights are tied, defaults to False
        """
        super(DenoisingAutoencoder, self).__init__()
        self.name = "DAE"
        self.network = network
        self.gain = gain
        # self.encoder = encoder
        # self.decoder = decoder
        # self.encoder = nn.Sequential(OrderedDict([(key, layer) for key, layer in encoder.named_children() if not key  in ["activation", "max_pool", "max_unpool"]]))
        # self.decoder = nn.Sequential(OrderedDict([(key, layer) for key, layer in decoder.named_children() if not key in ["activation", "max_pool", "max_unpool"]]))
        self.encoder = nn.Sequential(OrderedDict([(key, layer) for key, layer in encoder.named_children() if not key  in ["activation"]]))
        self.decoder = nn.Sequential(OrderedDict([(key, layer) for key, layer in decoder.named_children() if not key in ["activation"]]))
        self.corruption = torch.nn.Dropout(p=corruption) if isinstance(corruption, float) else corruption
        self._initialise_weight_bias(self.encoder.main.weight, self.encoder.main.bias, self.gain)
        self._initialise_weight_bias(self.decoder.main.weight, self.decoder.main.bias, self.gain)

    @staticmethod
    def _initialise_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        """
        Initialise the weights in a the Linear layers of the DenoisingAutoencoder.

        :param weight: weight Tensor of the Linear layer
        :param bias: bias Tensor of the Linear layer
        :param gain: gain for use in initialiser
        :return: None
        """
        if weight is not None:
            nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def copy_weights(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder and decoder should be instances of torch.nn.Linear.

        :param encoder: encoder Linear unit
        :param decoder: decoder Linear unit
        :return: None
        """
        try:
            encoder.main.weight.data.copy_(self.encoder.main.weight)
            encoder.main.bias.data.copy_(self.encoder.main.bias)
            decoder.main.weight.data.copy_(self.decoder.main.weight)
            decoder.main.bias.data.copy_(self.decoder.main.bias)
        except:
            pass

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        transformed = self.encoder(batch if self.network=="CNN" else torch.flatten(batch, 1))
        if self.corruption is not None:
            transformed = self.corruption(transformed)
        return transformed

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        if hasattr(self.decoder, "max_unpool"):
            setattr(self.decoder.max_unpool, "indices", self.encoder.max_pool.indices)
        transformed = self.decoder(batch)
        if self.corruption is not None:
            transformed = self.corruption(transformed)
        return transformed
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))
    
    def predict(self, batch: torch.Tensor, **kwargs) -> tuple:
        embedding = self.encode(batch)
        reconst = self.decode(embedding)
        return (reconst.detach(),
                batch.detach(),
                embedding.detach(), 
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
        inputs = dict(reconst=results[0].flatten(1), origin=results[1].flatten(1))
        if len(results)>2:
            inputs=dict(inputs, embedding=results[2])
        # if self.generative:
        #     inputs = dict(inputs, mu=results[-4], log_var=results[-3])
        
        loss = loss_caller(inputs, **kwargs)
        
        return loss
    