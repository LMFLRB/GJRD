import torch, os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import List, Optional
from collections import OrderedDict
from tensorboardX import SummaryWriter
from copy import deepcopy as copy

from Network.dae import DenoisingAutoencoder as DAE
from LossModule import Loss
from Trainer.metrics import Metric
from Trainer.utils import (get_optimizer, 
                               get_scheduler, 
                               update_callback, 
                               earlystop,
                               write_confumatrix,
                               visualize_embeddings,
                               save_clustered_image,
                               get_time,
                               get_version_numbers,
                               myEventLoader)

# compare_logdir=  "G:\MyCode\GCSD\compare"

class Experiment():
    def __init__(self, config) -> None:
        self.params = config
        self.cuda = config["cuda"]
        self.max_version = config["max_version"]
        self.corruption = config["corruption"]
        self.update_freq = config["update_freq"]
        self.silent = config["silent"]

        self.enable_earlystop = config["enable_earlystop"]
        self.delta_earlystop = config["delta_earlystop"]
        self.patience = config["patience"]

        self.enable_pretrain=config["enable_pretrain"]
        self.cluster_only=config["cluster_only"]
  
        self.pretrain_epochs = config["pretrain_epochs"]
        self.finetune_epochs = config["finetune_epochs"]
        self.cluster_epochs = config["cluster_epochs"]
        self.log_dir = config['log_dir']
        self.log_text_path = f"{self.log_dir}/{self.params['log_text_path']}.txt"
        self.save_results = config['save_results']
        
        self.data_params=copy(config["data"])
        self.batch_size= self.data_params.pop("batch_size")
        self.evaluate_batch_size=self.data_params.pop("evaluate_batch_size")
       

        self.optimizer = get_optimizer(config.get("optimizer"))
        self.scheduler = get_scheduler(config.get("scheduler"))
        self.cluster_optimizer = get_optimizer(config.get("cluster_optimizer"))
        self.cluster_scheduler = get_scheduler(config.get("cluster_scheduler"))

        self.loss_caller = Loss[config["loss"]["name"]](**config["loss"])
        self.loss_caller_ae = Loss["VAE"](**dict(weights=dict(reconst=1.0, regular=1.0e-4)))
        self.metric_caller = Metric(**config["metric"])

        self.cur_procedure = "SDAE"

        self.data_iter = lambda data_loader, postfix, desc, leave: \
                                tqdm(data_loader, 
                                     postfix=postfix,
                                     disable=self.silent,
                                     desc=desc,
                                     leave=leave)
        
        self.train_iter = lambda data_loader, postfix, leave: \
                                tqdm(data_loader,
                                     postfix=postfix,
                                     leave=leave,  
                                     desc='training loop',
                                     unit="batch",
                                     dynamic_ncols=True,)
        
        self.val_iter = lambda data_loader, postfix: \
                                tqdm(data_loader, 
                                     postfix=postfix,
                                     desc='validate loop',
                                     unit="batch",
                                     leave=False,  
                                     dynamic_ncols=True,
                                     )
    
    def set_version(self, log_dir:str=None, version=None):
        self.log_version=version
        self.logdir_cluster = f"{log_dir}/version_{version}"    
        if self.save_results:
            os.makedirs(f"{self.logdir_cluster}", exist_ok=True)  
            self.EventLoader = myEventLoader(self.logdir_cluster) 
    
    def events_to_mat(self):
        self.EventLoader.events_to_mat()
    
    def set_loss_function(self, loss_params:dict={}):
        self.loss_caller = Loss[loss_params.name](**loss_params)
        self.params.update(dict(loss=loss_params))

    def log_text(self, info: str="", silent: bool=False):
        os.makedirs(self.log_dir, exist_ok=True)
        def printf(func, args=(info)):
            def printf_line(info_line):
                cur_time = get_time()
                info_line = cur_time+': '+ (info_line if isinstance(info_line, str) else str(info_line))
                func(info_line.strip())
                if func.__name__=="write":
                    func("\n")
            if type(info) in [list, tuple]:
                for info_line in info:
                    printf_line(info_line)
            else:
                printf_line(info)
                    
        with open(self.log_text_path, "a") as f:
            printf(f.write,args=(info))
        if not silent:
            printf(print,args=(info))

    def pretrain_dae(self, model, dataset, val_dataset, logdir:str):
        current_dataset = dataset
        current_validat = val_dataset
        n_subautoencoders = len(model.dimensions)        
        self.enumerate_versions=n_subautoencoders
        self.cur_procedure = "DAE"
        self.logdir_sdae = logdir
        for index in range(n_subautoencoders):
            self.log_version=index
            encoder, decoder = model.get_stack(index)
            # manual override to prevent corruption for the last subautoencoder
            if index == (n_subautoencoders - 1):
                corruption = None
            else:
                corruption = copy(self.corruption)
            # initialise the subautoencoder
            sub_autoencoder = DAE(encoder, decoder, corruption, model.network,)
            logdir_cur=f"{logdir}/dae_sub_{index}"
            pretrained_ckpt=f"{logdir_cur}/pretrained.ckpt"

            if os.path.exists(pretrained_ckpt):
                sub_autoencoder.load_state_dict(torch.load(pretrained_ckpt))
                self.log_text(f"the {index+1}-th sub_autoencoder state_dict loaded from ckpt!")
                if self.cuda:
                    sub_autoencoder = sub_autoencoder.cuda()
                self.epoch=0     
                self.enumerate_epochs=self.pretrain_epochs      
            else:
                if self.cuda:
                    sub_autoencoder = sub_autoencoder.cuda()
                self.log_text(f"the {index+1}-th sub_autoencoder pretraining......")
                self.train_autoencoder(sub_autoencoder, 
                                       current_dataset, 
                                       current_validat,
                                       self.pretrain_epochs,
                                       log_dir=logdir_cur,
                                       corruption=None, # the sub_autoencoder already has dropout layer
                                       )
                self.log_text(f"the {index+1}-th sub_autoencoder pretrain finished!") 
            if index != (n_subautoencoders - 1):
                pred_out = self.predict(sub_autoencoder, current_dataset, batch_size=256)
                current_dataset = TensorDataset(pred_out[2], pred_out[-1])                                    
                if current_validat is not None:
                    pred_out = self.predict(sub_autoencoder, current_validat, batch_size=256)
                    current_validat = TensorDataset(pred_out[2], pred_out[-1])
            else:
                current_dataset = None  # minor optimisation on the last subautoencoder
                current_validat = None
            # copy the weights
            # sub_autoencoder.copy_weights(encoder, decoder)
        # DAE pretrain finished, set or SDAE
        self.cur_procedure = "SDAE"
        self.log_version = 0

    def train_autoencoder(self, model, dataset, 
                          val_dataset=None, 
                          epochs=None, 
                          log_dir:str=None,
                          corruption:float=0.0):
        
        corruption = self.corruption if corruption==0.0 else corruption
        
        writer = SummaryWriter(log_dir=log_dir)
        train_dataloader = DataLoader(dataset, self.batch_size, True, **self.data_params)
        iter_per_epoch=len(train_dataloader)
    
        loss_value = val_loss = -1.0
        self.enumerate_epochs=epochs if epochs is not None else self.finetune_epochs
        
        if epochs is None:
            self.enumerate_versions=1
            self.log_version=0
        else:
            self.log_version=int(log_dir.split("_")[-1])
        earlystop.count, earlystopped = 0, False
        
        optimizer = self.optimizer(model)
        scheduler = self.scheduler(optimizer)        
        model.train()
        # torch.save(model.state_dict(), os.path.join(compare_logdir, f"{self.cur_procedure}-{self.log_version}-integeral_init.ckpt"))
        for epoch in range(self.enumerate_epochs): 
            if self.enable_earlystop:
                earlystopped=earlystop(torch.tensor(val_loss), self.patience, self.delta_earlystop)                
            self.epoch = epoch
            data_iterator = self.train_iter(train_dataloader,
                                            postfix=OrderedDict(version=f"{self.log_version+1}/{self.enumerate_versions}",
                                                epoch=f"{epoch+1}/{self.enumerate_epochs}",
                                                loss="%.6f" % (loss_value or 0.0),
                                                val_loss="%.6f" % (val_loss or -1.0),
                                                ),
                                            leave=True if (epoch==self.enumerate_epochs-1 or earlystopped) else False,
                                            )       
            data_iterator.desc = f"{self.cur_procedure} {data_iterator.desc}"        
            for index, (batch, _) in enumerate(data_iterator):
                # torch.save(model.state_dict(), os.path.join(compare_logdir, f"{self.cur_procedure}-{self.log_version}-integeral.ckpt"))
                if self.cuda and not batch.is_cuda:
                    batch = batch.cuda(non_blocking=True)
                output = model(batch if corruption is None else F.dropout(batch, corruption))
                loss_dict = model.loss_function(
                                            (output,batch), 
                                            loss_caller=self.loss_caller_ae, 
                                            epoch=epoch, 
                                            is_training=True, 
                                            )
                loss = loss_dict["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                loss_value = float(loss.item())                
                if self.update_freq is not None and index % self.update_freq == 0:
                    data_iterator.set_postfix(OrderedDict(
                                        version=f"{self.log_version+1}/{self.enumerate_versions}",
                                        epoch=f"{epoch+1}/{self.enumerate_epochs}", 
                                        loss=f"{loss_value:.6f}", 
                                        val_loss=f"{val_loss:.6f}"))
                    loss_dict={f"{key}_step": value for key, value in loss_dict.items()}
                    update_callback(writer, epoch*iter_per_epoch+index, 
                                    dict(lr_step=optimizer.param_groups[0]["lr"], loss_step=loss_dict,),)
                    self.log_text(f"{data_iterator.desc}: version={self.log_version+1}/{self.enumerate_versions}, epoch={epoch+1}/{self.enumerate_epochs}, batch={index+1}/{iter_per_epoch}, loss={loss_value:.6f}", 
                                    silent=True)
                
            if scheduler is not None:
                scheduler.step()
            
            if self.update_freq is not None:
                if val_dataset is not None:                         
                    #val_output: (embed, recon, orig, label)
                    val_output = self.predict(model, val_dataset, self.batch_size, silent=True)[:2]# resonst and origin
                    val_loss_dict = model.loss_function(val_output, loss_caller=self.loss_caller, epoch=epoch, is_training=False)
                    val_loss = val_loss_dict["loss"]
                else:
                    val_loss_dict = dict(loss=-1.0)
                    val_loss = -1.0
                data_iterator.set_postfix(
                    OrderedDict(version=f"{self.log_version+1}/{self.enumerate_versions}",
                         epoch=f"{epoch+1}/{self.enumerate_epochs}", 
                         loss=f"{loss_value:.6f}", 
                         val_loss=f"{val_loss:.6f}"),
                )
                self.log_text(f"{data_iterator.desc} epoch end: version={self.log_version+1}/{self.enumerate_versions}, epoch={epoch+1}/{self.enumerate_epochs}, val_loss={val_loss:.6f}",
                            silent=True)
                val_loss_dict={f"val_{key}_epoch": value for key, value in val_loss_dict.items()}
                update_callback(writer, epoch, val_loss_dict)
            if earlystopped:
                self.log_text(f'\nEarly stopping @ epoch {epoch+1} as val_loss varing less than {self.delta_earlystop*100}% of the last for {self.patience} epochs.')
                break
        torch.save(model.state_dict(), f"{writer.logdir}/pretrained.ckpt")    
        writer.close()

    def train_cluster(self, model, dataset, val_dataset=None, log_dir:str=None, log_version=None): 
        if not hasattr(self, 'logdir_cluster') and log_dir is not None:
            self.logdir_cluster = log_dir     
        if not hasattr(self, 'log_version'):  
            self.log_version=get_version_numbers(self.logdir_cluster) if log_version is None else log_version
        
        if self.save_results:
            os.makedirs(f"{self.logdir_cluster}", exist_ok=True)   
            writer = SummaryWriter(log_dir=self.logdir_cluster)
            torch.save(model.state_dict(), f"{writer.logdir}/init.ckpt")
        self.cur_procedure="cluster"
        if not hasattr(self, "cluster_count_start") or \
              (hasattr(self, "cluster_model") and getattr(self, "cluster_model")!=model.name):
            self.cluster_model=model.name
        self.enumerate_versions=copy(self.max_version)
        self.enumerate_epochs=copy(self.cluster_epochs)

        val_dataloader = DataLoader(val_dataset, self.evaluate_batch_size, False, **self.data_params)  
        train_dataloader = DataLoader(dataset, self.batch_size, True, **self.data_params)        
    
        model.train()
        if model.name=="DEC":
            self.log_text("initializing cluster centers......")
            predicted, actual = model.init_cluster_centers(val_dataloader, self.cuda)
            self.log_text("clustering centers initialized!")   
        else:
            self.epoch=-1
            self.log_text("validating clustering of the initial/pretrained model...")   
            #output: (out, embed, recon, orig, label)
            output = self.predict(model, val_dataset, self.evaluate_batch_size, silent=False)  
            predicted, actual = output[0].max(1)[1].long(), output[-1]
            self.log_text("validation finished!")   
        predicted_previous = copy(predicted)
        acc, nmi, ari = model.metric_function(predicted, actual, self.metric_caller).values()
        delta_label = 1.0
        iters_per_epoch=len(train_dataloader)
        optimizer = self.cluster_optimizer(model)
        scheduler = self.cluster_scheduler(optimizer)
        
        earlystop.count, earlystopped = 0, False
        loss_value =acc_best=nmi_best=ari_best=0.0
        # acc_best_model=nmi_best_model=ari_best_model=None
        
        model.train()
        for epoch in range(self.cluster_epochs):
            self.epoch=epoch
            data_iterator = self.train_iter(
                                    train_dataloader,
                                    postfix=OrderedDict(
                                        version=f"{self.log_version+1}/{self.max_version}",
                                        epoch=f"{epoch+1}/{self.cluster_epochs}",
                                        acc="%.4f" % acc,
                                        nmi="%.4f" % nmi,
                                        ari="%.4f" % ari,
                                        loss="%.8f" % loss_value,
                                        dlb="%.4f" % delta_label,
                                        ),
                                    leave=True if (earlystopped or epoch==self.cluster_epochs-1) else False,
                                    )               
            data_iterator.desc = f"{self.cur_procedure} {data_iterator.desc}"    
            for index, (batch, actual) in enumerate(data_iterator):                
                # torch.save(model.state_dict(), os.path.join(compare_logdir, f"{self.cur_procedure}-{self.log_version}-integeral.ckpt"))
                if self.cuda and not batch.is_cuda:
                    batch = batch.cuda(non_blocking=True)
                output = model(batch)
                
                loss_dict = model.loss_function(output, 
                                           self.loss_caller,
                                           epoch=self.epoch, 
                                           is_training=True,
                                           cluster_only=self.cluster_only,
                                        #    enable=self.params.get('loss_enable')
                                           )
                
                loss = loss_dict["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                if self.update_freq is not None and index % self.update_freq == 0:
                    loss_value = float(loss.item())    
                    loss_dict["loss"] = loss_value      
                    assign=output[0] if isinstance(output, tuple) else output
                    metric_dict = model.metric_function(assign, actual, self.metric_caller)
                    acc, nmi, ari = metric_dict.values()
                    data_iterator.set_postfix(
                                        OrderedDict(                                        
                                            version=f"{self.log_version+1}/{self.max_version}",
                                            epoch=f"{epoch+1}/{self.cluster_epochs}",
                                            acc="%.4f" % acc,
                                            nmi= "%.4f" %  nmi,
                                            ari= "%.4f" %  ari,
                                            loss="%.4e" % loss_value,
                                            dlb="%.4f" % delta_label)
                                        )                    
                    self.log_text(f"{data_iterator.desc}: version={self.log_version+1}/{self.max_version}, epoch={epoch+1}/{self.cluster_epochs}, acc_step={acc:.4f}, nmi_step={nmi:.4f}, ari_step={ari:.4f}, loss_step={loss_value:.4e}, dlb_step={delta_label:.4f}",
                                silent=True)
                    metric_dict={f"{key}_step": value for key, value in metric_dict.items()}
                    loss_dict={f"{key}_step": value for key, value in loss_dict.items()}
                    step=epoch*iters_per_epoch+index
                    if self.save_results:
                        update_callback(writer, step, dict(lr=optimizer.param_groups[0]["lr"],
                                                       metric=metric_dict, 
                                                       loss=loss_dict))
                
            if scheduler is not None:
                scheduler.step()
            #output: (out, embed, recon, orig, label)
            output = self.predict(model, 
                                    val_dataset if val_dataset is not None else dataset, 
                                    self.evaluate_batch_size, 
                                    silent=True)
            pred, label = output[0], output[-1]
            loss_input=output[:-1]
            if model.name=="DEC":
                loss_dict = model.loss_function(loss_input, 
                                        self.loss_caller,
                                        epoch=self.epoch, 
                                        is_training=False)
            else: # seperate loss computation to saveral batches to avoid cases that out of memeory 
                loss_input_set = DataLoader(TensorDataset(*loss_input), self.evaluate_batch_size, False)
                loss_dict={}
                for index, loss_in_batch in enumerate(loss_input_set):
                    loss_dict_batch = model.loss_function(loss_in_batch, 
                                        self.loss_caller,
                                        epoch=self.epoch, 
                                        is_training=False)
                    if index==0:
                        loss_dict=loss_dict_batch
                    else:
                        loss_dict={key: loss_dict[key]+val for key,val in loss_dict_batch.items()}
                loss_dict = {key: val/(index+1) for key, val in loss_dict.items()}
            
            predicted = pred.max(1)[1].long()
            metric_dict = model.metric_function(predicted, label, self.metric_caller)
            acc, nmi, ari = metric_dict.values()
            delta_label = (predicted != predicted_previous).float().sum().item()/predicted.shape[0]
            predicted_previous = predicted
            if self.update_freq is not None:
                data_iterator.set_postfix(
                                        OrderedDict(                                        
                                            version=f"{self.log_version+1}/{self.max_version}",
                                            epoch=f"{epoch+1}/{self.cluster_epochs}",
                                            acc="%.4f" % acc,
                                            nmi= "%.4f" % nmi,
                                            ari= "%.4f" % ari,
                                            loss="%.4e" % loss_dict["loss"],
                                            dlb="%.4f" % delta_label)
                                        )
                self.log_text(f"{data_iterator.desc}: version={self.log_version+1}/{self.max_version}, epoch={epoch+1}/{self.cluster_epochs}, acc_epoch={acc:.4f}, nmi_epoch={nmi:.4f}, ari_epoch={ari:.4f}, loss_epoch={loss_value:.4e}, dlb={delta_label:.4f}\n\n",
                              silent=True)
                if self.save_results:
                    metric_dict={f"{key}_epoch" if val_dataset is None else f"val_{key}_epoch": value for key, value in metric_dict.items()}
                    loss_dict={f"{key}_epoch" if val_dataset is None else f"val_{key}_epoch": value for key, value in loss_dict.items()}                    
                    update_callback(writer, epoch, dict(metric=metric_dict, loss=loss_dict, val_dlb=delta_label) )
            if acc>acc_best:                
                acc_best = acc
                self.acc_best_ckpt=copy(model.state_dict())
                # torch.save(model.state_dict(), f"{writer.logdir}/acc_best.ckpt")    
            # if nmi>nmi_best: 
            #     nmi_best = nmi     
            #     self.nmi_best_ckpt=copy(model.state_dict() )         
            #     # torch.save(model.state_dict(), f"{writer.logdir}/nmi_best.ckpt")    
            # if ari>ari_best:
            #     ari_best = ari         
            #     self.ari_best_ckpt=copy(model.state_dict()  )                  
            #     # torch.save(model.state_dict(), f"{writer.logdir}/ari_best.ckpt")    
            
            if self.enable_earlystop and earlystop(torch.tensor(acc), self.patience, self.delta_earlystop):
                # self.log_text(f'Early stopping @ epoch {epoch+1} as predicted labels varing less than {self.delta_earlystop*100}% of the last for {self.patience} epochs.'
                # )
                self.log_text(f'Early stopping @ epoch {epoch+1} as predicted predicted accuracy varing less than {self.delta_earlystop*100}% for {self.patience} epochs.'
                )
                break
        try:
            if self.save_results:
                torch.save(self.acc_best_ckpt, f"{writer.logdir}/acc_best.ckpt")  
                # torch.save(nmi_best_ckpt, f"{writer.logdir}/nmi_best.ckpt")    
                # torch.save(ari_best_ckpt, f"{writer.logdir}/ari_best.ckpt")    
                torch.save(model.state_dict(), f"{writer.logdir}/last.ckpt")    
        except:
            pass
          
        self.log_text(f"Clustering training finished!")
        
    def evaluate(self, model, dataset, save_results:bool=True, visualize:bool=False,
                 log_dir_cluster:str="", 
                 ckpt_path:str="",
                 ckpt_mode:str="last",
                 **kwargs) -> dict:
        
        if not hasattr(self, 'logdir_cluster'):
            self.logdir_cluster = log_dir_cluster
        if ckpt_mode in ["acc", "nmi", "ari"]:  
            if ckpt_path=="":
                ckpt_path=f"{self.logdir_cluster}/{ckpt_mode}_best.ckpt"
                model.load_state_dict(torch.load(ckpt_path))
            else:
                pass
            mode="best"
            self.log_text(f"loading pretrained best {ckpt_mode} autoencoder")
        else:
            try:
                ckpt_path=f"{ckpt_path}/last.ckpt"
                model.load_state_dict(torch.load(ckpt_path))
            except:
                pass
            mode="last"
            self.log_text(f"loading pretrained last autoencoder")

        self.epoch=0
        self.enumerate_epochs=1
        self.enumerate_versions=2
        ver_num = self.log_version=get_version_numbers(self.logdir_cluster)       
        
        batch_size=self.evaluate_batch_size if kwargs.get("batch_size") is None \
                                            else kwargs["batch_size"]
        #output: (out, embed, recon, orig, label)
        output = self.predict(model, dataset, batch_size)
        predicted = output[0].max(1)[1].detach().cpu().numpy()
        truelabel = output[-1].cpu().numpy()
        metrics = model.metric_function(predicted, truelabel, self.metric_caller)
        loss_name=self.loss_caller.name
        if hasattr(self.loss_caller, 'ddc1'):
            if hasattr(self.loss_caller.ddc1, 'entropy_order'):
                loss_name=loss_name+f'-{self.loss_caller.ddc1.entropy_order:.0f}'
        self.log_text(f"Clustering results of the {ver_num+1}-th validation with {model.name}-{loss_name} on {dataset.name} in {mode}-mode:\n\t {metrics}")
        if save_results and self.save_results:
            # write_results_to_txt(f"{self.logdir}/MontoCarloResultsItem_{mode}.txt", metrics, count=ver_num, mode="a")   
            if dataset.is_image and not self.params.use_processed_data:  
                save_clustered_image(output[-2], truelabel, predicted, f"{self.logdir_cluster}/images_{mode}")
            if visualize:
                file_tag="{}-{}-{}".format(dataset.name, loss_name, 'Enc' if model.encode_only else 'AE')                        
                write_confumatrix(predicted, 
                                    truelabel, 
                                    clusters=dataset.clusters,
                                    filepath=f"{self.logdir_cluster}/{file_tag}_confusion_matrix_{mode}", 
                                    )
                visualize_embeddings(output[1].detach().cpu().numpy(), 
                                        predicted, 
                                        truelabel, 
                                        f"{self.logdir_cluster}/{file_tag}_latent_visualization_{mode}")
         
        return metrics
    
    def predict(self, model, dataset, batch_size: int=1024, silent=True, **kwargs):
        dataloader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size, False, **self.data_params)
        data_iterator = tqdm(dataloader, 
                            disable=silent,
                            desc=f"{self.cur_procedure} validate loop",
                            unit="batch",
                            leave=True,  
                            dynamic_ncols=True,
                            postfix=OrderedDict(
                                    version=f"{self.log_version+1}/{self.enumerate_versions}", 
                                    epoch=f"{self.epoch+1}/{self.enumerate_epochs}"), )
        
        truelabel, modelout = [], []   
        model.eval()        
        with torch.no_grad():
            for _, (feature, label) in enumerate(data_iterator):
                if self.cuda and not feature.is_cuda:
                    feature = feature.cuda(non_blocking=True)
                truelabel.append(label.detach())
                modelout.append(model.predict(feature))
        Res = ()
        for i in range(len(modelout[0])):
            Res = Res + (torch.cat([modelout[j][i] for j in range(len(modelout))]),)
            
        model.train()
        return Res + (torch.cat(truelabel),) #output: (out, embed, recon, orig, label)
        