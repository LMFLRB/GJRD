'''
Author: Mingfei Lu
Description: 
Date: 2022-04-08 17:00:00
LastEditTime: 2022-09-15 09:34:37
'''
from .loss import *

Loss = {'MSE': myMSE,
        'CE': CrossEntropy,
        'KLD': KLDLoss,
        'CSD': CSDLoss,
        'GJRD': GJRDLoss,
        'PJRD': PJRDLoss,
        'DDC': DDCLoss,
        "DEC": DECLoss,
        "VAE": VAELoss,
        "KMeans": KMeans,
        }