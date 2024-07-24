import numpy as np
import torch
import os
import yaml
import time
from easydict import EasyDict as Edict
from typing import Union

from scipy.io import savemat, loadmat
import pandas as pd
from DataModule import LoadDataset

from Network import Models
from Trainer import (Experiment, 
                        make_model_data_consistent,
                        transform_to_edict,
                        easydict_to_dict,
                        make_dict_consistent,
                        manual_seed_all,
                        remove_path,
                        copy
                        )

cur_dir = os.path.dirname(__file__)
     
def pre_train_ae(data, loss, config:dict={}):
    if not config.experiment.enable_pretrain or config.model.encode_only:
        return 
    else:
        exp_params  = config.experiment
        model_params= config.model 
        data_params = config.data 
        ds_train, ds_val = data   

        log_dir_dae=f"{exp_params.log_dir}/autoencoder"
        log_dir_sdae=f"{log_dir_dae}/sdae"                    
        ckpt_path_sdae=f"{log_dir_sdae}/pretrained.ckpt"
        if not os.path.exists(ckpt_path_sdae):              
            exp_params.save_results=True
            os.makedirs(f"{log_dir_sdae}", exist_ok=True)
            exp_params.log_text_path = f"ae_pre_textlog"
            experiment = Experiment(exp_params)
            experiment.log_text([f"configurations loaded for {loss} from {config.file}",
                                f"configurating experiment for stackVAE on {data_params.name}...",])  
            make_model_data_consistent(ds_train, model_params.autoencoder)         
            model = Models[model_params.name](**model_params)
            autoencoder = model.encoder
            if exp_params.cuda:
                autoencoder.cuda()
            experiment.log_text([f"Autoencoder substacked pretraining stage."])
            experiment.pretrain_dae(autoencoder,ds_train,ds_val,log_dir_dae)
            experiment.log_text("Autoencoder training stage.")
            experiment.train_autoencoder(autoencoder,ds_train,ds_val,log_dir=log_dir_sdae)
        

        return ckpt_path_sdae

def find_best_seeds(config, tag_sel:str="ACC", log_dir: str="", results_file:str="montecarlo.csv", finetune_top_k:int=None):
    # find seeds producing the best performances
    if not tag_sel in ["ACC", "NMI", "ARI"]:
        ValueError("Please input the right tag for performance")
    root_dir = config.experiment.log_dir if len(log_dir)==0 else log_dir
    finetune_top_k=config.experiment.n_finetune if finetune_top_k is None else finetune_top_k
    try:
        Seeds_sel=pd.read_csv(f"{root_dir}/{results_file}").sort_values(by='acc', ascending=False)['seed'].values[:finetune_top_k].tolist()     
    except:
        Seeds_sel=[]
        pass
    return Seeds_sel

def fit_with_seeds(data, loss, config:dict={}, Seeds:Union[dict, list]=None, fit_mode:str="montecarlo", retrain=False):       
    exp_params  = config.experiment
    model_params= config.model 
    data_params = config.data 
    if fit_mode=="montecarlo":
        exp_params.cluster_epochs = 5
        exp_params.save_results = False # do not save resutls when conducting montecarlo experiments
    elif fit_mode in ["finetune", "seed_eval"]:
        exp_params.cluster_epochs = 100
        exp_params.save_results = True # save resutls when conducting montecarlo experiments
    elif fit_mode=="eval_only":
        exp_params.cluster_epochs = 100
        exp_params.save_results = True # save resutls when conducting montecarlo experiments
        finetune_done=os.path.exists(os.path.join(exp_params.log_dir, "finetune-done"))
        exp_params.eval_only = True and finetune_done 
        eval_dir = f"{exp_params.log_dir}/{'finetune' if finetune_done else 'eval_only'}_cluster"

    exp_params.log_text_path = f"{fit_mode}_textlog"
    
    ds_train, ds_val = data
    make_model_data_consistent(ds_train, model_params.autoencoder)
    
    root_dir = exp_params.log_dir

    filename_results=f"{root_dir}/{fit_mode}.csv"
    cluster_dir = f"{root_dir}/{fit_mode}_cluster" 
    if retrain:
        if os.path.exists(filename_results):
            os.remove(filename_results)
        if os.path.exists(cluster_dir):
            remove_path(cluster_dir)

    seeds=Seeds[loss] if isinstance(Seeds, dict) else Seeds
    exp_params.max_version = len(seeds)
    for version, seed in enumerate(seeds):
        # make the model repruducible
        exp_params.seed = seed
        manual_seed_all(seed)                

        # version=get_version(cluster_dir)
        log_dir_cluster=f"{cluster_dir}/version_{version}"
        # exp_params.log_dir = root_dir
        if exp_params.save_results:
            os.makedirs(f"{log_dir_cluster}", exist_ok=True)
        
        experiment = Experiment(exp_params)
        experiment.log_text([f"generated seed={seed}",
                            f"configurations loaded for {loss} from {config.file}",
                            f"configurating experiment for {data_params.name} with {model_params.name}-{model_params.autoencoder.network}-{'FC' if model_params.encode_only else 'AE'}...",])
    
        model = Models[model_params.name](**model_params)
        
        experiment.log_text(f"configurating model...")          
        if exp_params.enable_pretrain and not model_params.encode_only:
            ckpt_path_sdae = pre_train_ae(data, loss, config)
            model.encoder.load_state_dict(torch.load(ckpt_path_sdae))                    
        if exp_params["cuda"]:
            model.cuda()
        
        df_dict=dict(seed=int(seed))
        
        if fit_mode in ["montecarlo", "finetune"] or not exp_params.eval_only:
            experiment.log_text(f"{model.name} clustering with {loss} on {data_params.name} {fit_mode} training stage.")
            experiment.log_text(f"{experiment.loss_caller.weights}")
            experiment.train_cluster(model, ds_train, ds_val, log_dir_cluster)
        # for mode in ["last","acc"]:
        for mode in ["last" if fit_mode=="montecarlo" else "acc"]:
            metric_mode=experiment.evaluate(model, ds_val, 
                                        ckpt_mode=mode, 
                                        log_dir_cluster=log_dir_cluster,
                                        visualize=True, )
            df_dict.update({key if mode=="last" else f"{key}_best": val for key,val in  metric_mode.items()})
        # create a DataFrame and then add it to the end of the file
        data_to_append = pd.DataFrame(df_dict, columns=df_dict.keys(), index=[version])
        data_to_append.to_csv(filename_results, mode='a', header=False if os.path.exists(filename_results) else True, index_label="version")  

        if os.path.exists(log_dir_cluster):
            with open(f"{log_dir_cluster}/config.yaml", 'w') as f:
                yaml.dump(easydict_to_dict(config), f)
            experiment.log_text(f"saving current configurations to 'config.yaml'. ")

        del model, experiment
    
    with open(os.path.join(f"{root_dir}", f'{fit_mode}-done'), 'w') as f:
        f.write('done')
        
if __name__ == "__main__":    
    with open(f'configs.yaml', 'r') as file:
        config = transform_to_edict(yaml.safe_load(file)) 
    os.makedirs(config.experiment.log_dir, exist_ok=True)  
    config.file='configs.yaml'
    root_dir = copy(config.experiment.log_dir)
    # root_dir = "Debug"
    loss = config.experiment.loss.name
    for dataset in ["MNIST", "FashionMNIST", "STL10"]:#
        config.data.cuda =  config.experiment.cuda = torch.cuda.is_available()#  
        config.data.name = dataset
        for resnet_type in ["resnet18", "resnet50"]:            
            config.experiment.resnet_type=resnet_type
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
                if not os.path.exists(os.path.join(config.experiment.log_dir, 'montecarlo-done')):
                    montecarlo_seeds=torch.randperm(10000).tolist()[:config.experiment.n_montecarlo] 
                    fit_with_seeds(data, loss, config, montecarlo_seeds, fit_mode="montecarlo", retrain=True)
                else:
                    print("montecarlo-done before")
                # if not os.path.exists(os.path.join(config.experiment.log_dir, 'finetue-done')):
                #     finetune_seeds = find_best_seeds(config, finetune_top_k=config.experiment.n_finetune) 
                #     fit_with_seeds(data, loss, config, finetune_seeds, fit_mode="finetune", retrain=False)
                # else:
                #     print("finetue-done before")
                finetune_seeds = find_best_seeds(config, finetune_top_k=config.experiment.n_finetune) 
                fit_with_seeds(data, loss, config, torch.randperm(10000).tolist()[:2] , fit_mode="finetune", retrain=False)
            
            del data
    
