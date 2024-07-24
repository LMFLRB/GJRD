import torch
import os
import yaml

import pandas as pd
from DataModule import LoadDataset

from Network import Models
from Trainer import (Experiment, 
                        make_model_data_consistent,
                        transform_to_edict,
                        manual_seed_all,
                        copy
                        )

cur_dir = os.path.dirname(__file__)
 
def eval(data, loss, config:dict={}):       
    exp_params  = config.experiment
    model_params= config.model 
    data_params = config.data 
    exp_params.max_version = 1

    ds_train, ds_val = data
    make_model_data_consistent(ds_train, model_params.autoencoder)

    fit_mode = 'finetune'    
    root_dir = exp_params.log_dir
    eval_dir = f"{exp_params.log_dir}/{fit_mode}_cluster"
    exp_params.log_text_path = f"{fit_mode}_textlog"
    

    version=0
    filename_results=f"{root_dir}/{fit_mode}.csv"
    cluster_dir = f"{root_dir}/{fit_mode}_cluster" 
    log_dir_cluster=f"{cluster_dir}/version_{version}"
    # exp_params.log_dir = root_dir
    if exp_params.save_results:
        os.makedirs(f"{log_dir_cluster}", exist_ok=True)
    
    saved_ckpt=torch.load(f"Pretrained/{'_'.join(root_dir.split(os.sep)[1:])}.ckpt")
    # seeds=dict(MNIST=2803,FashionMNIST=452,STL10=6713)
    # torch.save(dict(state_dict=saved_ckpt, seed=seeds[data_params.name]),
    #            f"Pretrained/{'_'.join(root_dir.split(os.sep)[1:])}.ckpt")
    seed=exp_params.seed = saved_ckpt['seed']
    df_dict=dict(seed=int(seed))
    manual_seed_all(seed)     
    experiment = Experiment(exp_params)
    experiment.log_text([f"generated seed={seed}",
                        f"configurations loaded for {loss} from {config.file}",
                        f"configurating experiment for {data_params.name} with {model_params.name}-{model_params.autoencoder.network}-{'FC' if model_params.encode_only else 'AE'}...",
                        f"configurating model..."])
    model = Models[model_params.name](**model_params)  
    model.load_state_dict(saved_ckpt['state_dict'])
    if exp_params["cuda"]:
        model.cuda()
    metric_mode=experiment.evaluate(model, ds_val, log_dir_cluster=log_dir_cluster, ckpt_mode="acc", ckpt_path=None,)
    df_dict.update({f"{key}_best": val for key,val in  metric_mode.items()})
    # create a DataFrame and then add it to the end of the file
    data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys(), index=[version])
    data_to_append.to_csv(filename_results, mode='a', header=True if (version==0 or not os.path.exists(filename_results)) else False, index_label="version")  


if __name__ == "__main__":    
    with open(f'configs.yaml', 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    os.makedirs(config.experiment.log_dir, exist_ok=True)  
    config.file='configs.yaml'
    root_dir = copy(config.experiment.log_dir)
    loss = config.experiment.loss.name
    for dataset in ["STL10", "FashionMNIST", "MNIST"]:#
    # for dataset in ["STL10"]:   
        config.data.cuda =  config.experiment.cuda = torch.cuda.is_available()#  
        config.data.name = dataset     
        config.experiment.resnet_type="resnet50" if dataset=="STL10" else "resnet18"
        data = LoadDataset(config.data, 
                        config.experiment.use_processed_data,
                        config.experiment.feature_type,
                        config.experiment.resnet_type
                        )  #dataset  
        ###########################################################################################
        for framework,encode_only in zip(["AE", "FC"], [False, True]):
            config.model.encode_only = encode_only                   
            config.model.use_processed_data=config.experiment.use_processed_data
            config.model.resnet_type=config.experiment.resnet_type
            config.model.feature_type=config.experiment.feature_type 
            config.model.autoencoder.network="MLP" if config.experiment.feature_type=="linear" else "CNN"
                    
            config.experiment.log_dir = os.path.join(root_dir,dataset,
                            f"{config.experiment.resnet_type}-{config.experiment.feature_type}",
                            f"{config.model.name}-{config.model.autoencoder.network}-{framework}",
                            loss)   
            eval(data, loss, config)
        
        del data
    
