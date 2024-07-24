'''
Author: Mingfei Lu
Description: 
Date: 2022-02-16 10:17:37
LastEditTime: 2022-11-14 17:17:10
'''
import os
import sys
import torch
import numpy as np
import time
import random
from logging import warning
from typing import Any, List, Union
from scipy.io import savemat
from shutil import rmtree
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision import transforms
from typing import Optional
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import seaborn as sns
import uuid
import matplotlib.pyplot as plt

from copy import deepcopy as copy

from DataModule import (ordered_with_cluster,
                        set_label_batch_balanced)

tb_started=False

def is_power_of_2(n: int) -> bool:        
        """
        Given input number n, returns bool of whether n is power of number 2.
        """
        if n==0:
            return True
        return (n & (n-1)) == 0

def is_power_of(n, m):
    from torch import log, tensor, equal, floor, float

    if n <= 0 or m <= 0:
        return False
    k = log(tensor(n, dtype=float)) / log(tensor(m, dtype=float))
    return equal(k, floor(k))
        
def clear_all():
    for key in list(globals().keys()):
     if (not key.startswith("__")) and (key != "key"):
         globals().pop(key) 
         print(key)
    del key

def transform_to_edict(in_dict):
    in_dict=edict(in_dict)
    for key, value in in_dict.items():
        if isinstance(value, dict):
            in_dict[key]=transform_to_edict(value)
    return in_dict  

def easydict_to_dict(ed_obj):
    if not isinstance(ed_obj, edict):
        return ed_obj
    normal_dict = dict(ed_obj)
    for key, value in normal_dict.items():
        if isinstance(value, edict):
            normal_dict[key] = easydict_to_dict(value)
    return normal_dict

def make_dict_consistent(dest_dict, source_dict):
    for key, value in source_dict.items():
        if isinstance(value, dict):
            make_dict_consistent(dest_dict[key], value)
        else:
            dest_dict[key]=value
            
def cat_listed_dict(in_list):
    def cat_dict_to_list(in_dict, out_dict, initial=True):    
        for key, value in in_dict.items():
            if isinstance(value, dict):
                if initial:
                    out_dict[key]={}                
                cat_dict_to_list(value, out_dict[key], initial)
            else:
                value = value.detach().cpu().numpy() if type(value)==torch.Tensor else np.array(value)            
                if len(value.shape)<1:
                    value = np.expand_dims(value, axis=0)
                if initial:
                    out_dict[key]=[value] 
                else:
                    out_dict[key].append(value) 
    def cat_dict_of_list_to_numpy(in_dict):
        out_dict={}
        for key, value in in_dict.items():
            if isinstance(value, dict):
                out_dict[key]=cat_dict_of_list_to_numpy(value)
            else:
                out_dict[key]=np.vstack(value)
        return out_dict
    
    out_dict={}
    for num, data_dict in enumerate(in_list):
        cat_dict_to_list(data_dict, out_dict, num==0)
    return cat_dict_of_list_to_numpy(out_dict)

def str_repeat(str, n, **kwagrs):
    ## repeat the first m letters of the given string n times     
    mode = 'list' if kwagrs.get("mode") is None else kwagrs["mode"]
    front_len = len(str) if (kwagrs.get("m") is None  or kwagrs['m'] > len(str)) else kwagrs['m']   

    front = str[:front_len]
    result = '' if mode=='cat' else []    
    for i in range(n):
        if mode=='cat':
            result = result + front
        else:
            result.append(front)
    return result

def get_ckpt(tb_logger, load_version: int=-1, style: str = 'train' ):    
    log_version = tb_logger.version-1 if style == 'train' else tb_logger.version    
    version = log_version if load_version==-1 else load_version

    ckpt_path = os.path.join(tb_logger.root_dir, f"version_{version}", "checkpoints")    
    ckpt_last = get_file(ckpt_path, format='.ckpt', part_name='last' )[0]
    ckpt_best = get_file(ckpt_path, format='.ckpt', part_name='epoch')[0]
    ckpt_init = get_file(ckpt_path, format='.ckpt', part_name='init_dict')[0]
    ckpt=dict(last=ckpt_last[-1] if (len(ckpt_last)>0) else None,
              best=ckpt_best[-1] if (len(ckpt_best)>0) else None,
              init=ckpt_init[-1] if (len(ckpt_init)>0) else None,)
    return ckpt

def copy_files(source_path, destination_path):
    from shutil import copy, copytree
    os.makedirs(destination_path, exist_ok=True)
    # 获取源路径下的所有文件和文件夹
    file_list = os.listdir(source_path)
    # 遍历文件列表
    for file_name in file_list:
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        
        # 如果是文件则进行拷贝操作
        if os.path.isfile(source_file) and not os.path.exists(destination_file):
            copy(source_file, destination_file)
            
        # 如果是文件夹则递归调用函数进行拷贝
        elif os.path.isdir(source_file) and not os.path.exists(destination_file):
            copytree(source_file, destination_file)

def remove_path(file_dir):
    rmtree(file_dir)

def get_file(file_dir, part_name: str=r"mypart", format: str=r".myformat"):
    # find files in file_dir with part_name or format style
    myfiles, dirs = [], []
    try:
        for root, dirs, files in os.walk(file_dir):  
            for file in files:  
                filename, fileformat = os.path.splitext(file)
                if not format == r".myformat" and not part_name==r"mypart":
                    if  fileformat == format and part_name in filename:  # find the files with assigned format and part name
                        myfiles.append(os.path.join(root, file))  
                else:
                    if  fileformat == format or part_name in filename:  # find the files with assigned format or part name
                        myfiles.append(os.path.join(root, file))  
    except:
        warning("failed to get file with the give format or part name !")
    return myfiles, dirs if len(dirs)>0 else [file_dir]

def set_differed_filename(file_dir, constpart: str=r"mypart", format: str=r".myformat"):
    num_list = []
    num_cur = 0 
    file_list = get_file(file_dir, constpart, format)[0]
    if len(file_list)==0:
        # filename=constpart+str(num_cur=0)+format
        num_cur=0
    else:
        for file in file_list:
            start_pos = file.find(constpart)
            end_pos   = file.find(format)
            if start_pos+len(constpart)<end_pos:
                num = int(file[start_pos+len(constpart)+1:end_pos])  
            else:
                num = 0
            num_list.append(num)
        if num_cur in num_list:
            num_cur = max(num_list) + 1

    return  os.path.join(file_dir, f"{constpart}_{num_cur}{format}")  # file_dir+constpart+str(num_cur)+format
        
def cvtColor(colorImage, cvtType, flag=False):
    if flag:
        import cv2        
        from PIL import Image
        import numpy as np
        if isinstance(colorImage, torch.Tensor):
            img_cv2 = colorImage.cpu().numpy() if colorImage.device.type=='cuda' else colorImage.numpy()
        else:
            img_cv2 = colorImage

        flag_normalize, flag_transpose, new_shape = False, False, []
        if (img_cv2.shape[-3]==3 or img_cv2.shape[-3]==1):
            img_cv2 = img_cv2.transpose([0,2,3,1])
            flag_transpose=True

        if img_cv2.max()<1.01 and img_cv2.min()<-0.01:
            img_cv2 = np.clip((img_cv2+1)/2*255+0.5,0,255)
            flag_normalize=True
        elif img_cv2.max()<1.01 and img_cv2.min()>-0.01:
            img_cv2 = np.clip((img_cv2)*255+0.5,0,255)
            flag_normalize=True
        img_cv2 = np.stack(tuple([cv2.cvtColor(np.uint8(img_cv2[i]), cvtType) for i in range(img_cv2.shape[0])]))

        if flag_normalize:
            img_cv2=(img_cv2/255.0-0.5)*2.0
        if flag_transpose:
            img_cv2=img_cv2.transpose([0,3,1,2])
        
        
        return torch.from_numpy(img_cv2).to(colorImage.device)
    else:
        return colorImage

def image_visualize(image_data, title=""):
    from PIL import Image
    import numpy as np
    import torch

    # if isinstance(image_data, torch.Tensor):  
    #     if max(image_data) < 1.2:  
    #         image_data = image_data.mul(255).add_(0.5)
    #     image_data = image_data.clamp_(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    
    # elif isinstance(image_data, np.ndarray):           
    #     if max(image_data) < 1.2:  
    #         image_data = image_data*255+0.5
    #     image_data = np.transpose(np.clip(image_data, 0, 255), (1, 2, 0)).astype(np.uint8)
             
    if image_data.max() < 1.2:  
        image_data = image_data*255+0.5
    image_data = image_data.clamp_(0, 255).permute(1, 2, 0).numpy() if isinstance(image_data, torch.Tensor) else np.transpose(np.clip(image_data, 0, 255), (1, 2, 0))
    
    image = Image.fromarray(image_data.astype(np.uint8))   
    image.show(title)

    return image

def image_visualize_grid(img_tsr, *args, **kwargs):
    import torch
    from torchvision.transforms import ToPILImage
    from torchvision.utils import make_grid
    from PIL import Image
    import numpy as np

    if type(img_tsr) is list:
        if len(img_tsr[0].shape) == 4:
            img_tsr = torch.cat(tuple(img_tsr), dim=0)
        elif len(img_tsr[0].shape) == 3:
            img_tsr = torch.stack(tuple(img_tsr), dim=0)        
        img_tsr = make_grid(img_tsr.cpu(), *args, **kwargs)
    elif isinstance(img_tsr, np.ndarray):     
        img_tsr = make_grid(img_tsr.cpu(), *args, **kwargs)
        
    elif isinstance(img_tsr, torch.Tensor):
        img_tsr = make_grid(img_tsr, **kwargs)
        if img_tsr.max() < 1.2:
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            img_tsr = img_tsr.mul(255).add_(0.5)
        img_tsr = img_tsr.clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    PILimg = ToPILImage()(img_tsr)
    # PILimg.show()

    return PILimg

def save_imgrid(img_tsr, path, *args, **kwargs):
    from torchvision.transforms import ToPILImage
    from torchvision.utils import make_grid
    PILimg = ToPILImage()(make_grid(img_tsr.cpu(), *args, **kwargs))
    PILimg.save(path)
    return PILimg

def image_save_grid(img_tsr, path, show_img = False, *args, **kwargs):
    
    import torchvision.utils as vutils      
    vutils.save_image(img_tsr.cpu().data, path, *args, **kwargs)

    if show_img:
        from PIL import Image
        Image.open(path).show()

def image_add_white_edge(target, position='up_down'):
    def image_single_add(single_image, position):
        if position=='up_down':
            single_image[:,:2,:]  = torch.ones_like(single_image[:,:2,:])
            single_image[:,-2:,:] = torch.ones_like(single_image[:,-2:,:])
        else:
            single_image[:,:,:2]  = torch.ones_like(single_image[:,:,:2])
            single_image[:,:,-2:] = torch.ones_like(single_image[:,:,-2:])
            
    size = target.shape
    if len(size)==4:
        for num in range(size[0]):
            image_single_add(target[num,::], position)
    else:
        image_single_add(target, position)

def get_kernelsize(features: torch.Tensor, selected_num: int=10):
    ### estimating kernelsize with data
    from scipy.spatial.distance import pdist, squareform
    features_numpy = torch.flatten(features, 1).detach().cpu().float().numpy()
    k_features = squareform(pdist(features_numpy, 'euclidean'))
    return (np.sort(k_features, -1)[:, :selected_num]).mean((0,1)) 
  
def latent_perturb(orig_latent, perturb_num: int=20, perturb_lb=-1, perturb_ub=1, perturb_dim=-1, **kwargs):    
    from copy import deepcopy as copy
    dim = orig_latent.shape[0]
    if perturb_dim==-1:
        dim_0, dim_n = 0, dim
    else:
        dim_0, dim_n = perturb_dim, perturb_dim
    perturb_num = int(perturb_num)
    if not perturb_num%2==0:
        warning("perturbation quantity should be an even number, and here add one to it")
        perturb_num = perturb_num+1

    latent_grid = []
    for cur_dim in range(dim_0, dim_n, 1): 
        for cur_num in range(perturb_num+1):
            cur_perturb = perturb_lb+cur_num*(perturb_ub-perturb_lb)/perturb_num 
            cur_latent = copy(orig_latent)
            cur_latent[cur_dim] = cur_latent[cur_dim] + torch.tensor(cur_perturb, device=orig_latent.device) 
            latent_grid.append(cur_latent)

    latent_grid = torch.cat(tuple(latent_grid), 0)

    return latent_grid.reshape(-1, dim), perturb_num+1

def write_results_to_txt(file_Obj, Data_to_write, count:int=-1, mode:str='w'):
    if not hasattr(write_results_to_txt, "count"):
        write_results_to_txt.count=int(0)
    write_results_to_txt.count += 1
    write_count = count if count!=-1 else write_results_to_txt.count
    with open(file_Obj, mode) as f:
        f.write(f"write data the {write_count}-th time.\n")
        if isinstance(Data_to_write, (torch.Tensor, np.ndarray, List)):
            f.write(str(Data_to_write))
        elif isinstance(Data_to_write, dict):
            for index, (key, value) in enumerate(Data_to_write.items()):
                f.write(f"\t{key}: {value}\n")

def expand_dict(mydict:dict) -> dict:
    ex_dict={}
    for key, value in mydict.items():
        if isinstance(value, dict):
            ex_dict=dict(ex_dict, **expand_dict(value))
        else:
            ex_dict[key] = value
    return ex_dict

def update_callback(writer: SummaryWriter, iteration: int, update_dict: dict):
    paths=os.path.normpath(writer.logdir).split(os.sep)
    for key, value in expand_dict(update_dict).items():
        writer.add_scalar(tag=f"{paths[-1]}/{key}", 
                        scalar_value=value, 
                        global_step=iteration, 
                        display_name=key)
    # writer.add_scalars(main_tag=f"{paths[-2]}-{paths[-1]}", 
    #                    tag_scalar_dict=expand_dict(update_dict), 
    #                    global_step=iteration)

def make_model_data_consistent(data, model_params):
    # patch_size=data[0][0].shape    
    patch_size=data.feature_shape    
    if data.is_image:
        model_params["patch_size"] = patch_size[-2:]
        if model_params["network"] == "MLP":
            model_params["input_dim"] = np.array(patch_size).prod()
        elif model_params["network"] == "CNN":
            model_params["input_dim"] = patch_size[0]
    else:
        model_params["input_dim"] = patch_size[-1]
        if not model_params["network"] in ["CNN", "MLP"]: 
            model_params["latent_dim"]= model_params["hidden_size"]
    model_params["n_cluster"] = data.n_cluster
    # if not model_params['encode_only']: #nn.MaxUnpool() is not correctly used now in sdae model
    #     model_params['autoencoder']['use_maxpool']=False

def get_subfolders(folder_path):
    subfolders = []
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path): 
            file_path = os.path.join(folder_path, file_name)
            if os.path.isdir(file_path):
                subfolders.append(file_path)
                # subfolders.extend(get_subfolders(file_path))
    return subfolders

def get_immediate_subfolders(folder_path):
    try:
        subfolders = next(os.walk(folder_path))[1]
    except:
        subfolders = []
    subfolder_paths = [os.path.join(folder_path, subfolder) for subfolder in subfolders]
    return subfolder_paths

def get_version(path, partname="version_"):
    # subfolders=get_immediate_subfolders(path)
    subfolders=get_subfolders(path)
    if len(subfolders)==0:
        return 0
    else:
        versions=[]
        for name in subfolders:
            name = name.split("/")[-1]
            if partname in name:
                versions.append(int(name[(name.find(partname)+len(partname)):]))
        if len(versions)==0:
            max_version=0
        else:
            max_version=max(versions)+1
        return max_version

def get_version_numbers(path: str, part:str="version_", path_lever: Union[int,List]=-1) -> int:
    paths=os.path.normpath(path).split(os.sep)
    numbers=[]
    for lever in (path_lever if isinstance(path_lever, list) else [path_lever]):
        string=paths[lever]
        numbers.append(int(string[string.find(part)+len(part):]))
    return numbers[0] if len(numbers)==1 else numbers

def get_optimizer(optimizer):    
    if isinstance(optimizer, dict):
        optim_func = optimizer['name']
        optimizer_params = copy(optimizer)
        del optimizer_params['name']
        optimizer_ = lambda model: getattr(optim, 
                                       optim_func)(model.parameters(), **optimizer_params)
    else:
        optimizer_  = optimizer
    return optimizer_
    
def get_scheduler(scheduler):
    if isinstance(scheduler, dict):
        sched_func = scheduler['name']
        scheduler_params = copy(scheduler)
        del scheduler_params['name']
        scheduler_ = lambda optimizer: getattr(sched, 
                        sched_func)(optimizer, 
                                    **scheduler_params) if scheduler is not None else None
    else:
        scheduler_ = scheduler
    return scheduler_

# def write_confumatrix(predicted: np.ndarray, 
#                       truelabel: np.ndarray, 
#                       n_cluster: int, 
#                       filepath: str, 
#                       clusters: list = None,
#                       mode:str=None):
#     from .metrics import cluster_accuracy
#     import matplotlib.pyplot as plt
#     reassignment, _ = cluster_accuracy(truelabel, predicted, n_cluster)
#     predicted_reassigned = [
#         reassignment[item] for item in predicted
#     ]  # TODO numpify
#     confusion = confusion_matrix(truelabel, predicted_reassigned)
#     normalised_confusion = (
#         confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
#     )
#     os.makedirs(filepath,exist_ok=True)
#     confusion_id = uuid.uuid4().hex if mode is None else str(mode)
#     fig = plt.figure()  # 创建新的 Figure 对象
#     sns.heatmap(normalised_confusion, 
#                 xticklabels=clusters,
#                 yticklabels=clusters,
#                 cmap='Blues',
#                 ax=fig.gca()).get_figure().savefig(
#         f"{filepath}/confusion_{confusion_id}.png" 
#     )
#     plt.close(fig)  # 关闭 Figure 对象
#     savemat(f"{filepath}/confusion_matrix.mat", dict(CMat=confusion, 
#                                                      nCMat=normalised_confusion,
#                                                      clusters=clusters,
#                                                      labels=truelabel, 
#                                                      predLabels=predicted))
#     return confusion_id
# def write_confumatrix(predicted: np.ndarray, 
#                       truelabel: np.ndarray, 
#                       n_cluster: int, 
#                       filepath: str, 
#                       clusters: list = None):
#     from .metrics import cluster_accuracy
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     reassignment, _ = cluster_accuracy(truelabel, predicted, n_cluster)
#     predicted_reassigned = [
#         reassignment[item] for item in predicted
#     ]  # TODO numpify
#     confusion = confusion_matrix(truelabel, predicted_reassigned)
#     normalised_confusion = (
#         confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
#     )
#     confusion_id = uuid.uuid4().hex if filepath is None else filepath.split(os.sep)[-1]
    
#     # Create a mask to hide non-diagonal elements
#     annot = np.zeros_like(normalised_confusion).astype(str)
#     np.fill_diagonal(annot, np.round(normalised_confusion.diagonal(), 2))  # 对角线上填入数值
#     annot[annot == '0.0'] = ''  # 非对角线上设置为空字符串

    
#     fig = plt.figure()  # Create a new Figure object
#     ax = fig.gca()  # Get the current Axes instance
#     sns.heatmap(normalised_confusion, 
#                 xticklabels=clusters,
#                 yticklabels=clusters,
#                 cmap='Blues',
#                 annot=annot,  # Display the values on the heatmap
#                 fmt="",  # Format the values to 2 decimal places
#                 ax=ax)
#     # ax.set_xlabel('Predicted')
#     # ax.set_ylabel('True')
#     # ax.set_title('Confusion Matrix')
#     # fig.savefig(f"{filepath}.png") 
#     fig.savefig(f"{filepath}.pdf") 
#     plt.close(fig)  # Close the Figure object
#     # savemat(f"{filepath}.mat", dict(CMat=confusion, 
#     #                                                  nCMat=normalised_confusion,
#     #                                                  clusters=clusters,
#     #                                                  labels=truelabel, 
#     #                                                  predLabels=predicted))
#     return confusion_id

# def write_confumatrix_matlib(label_pred, label_true, title="Confusion Matrix", label_name=None, pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            if j==i:
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

# def visualize_embeddings(latents, labels, true_labels, file="latent_visualization"):
#     from sklearn.manifold import TSNE
#     # Perform s-tne dimensionality reduction
#     tsne = TSNE(n_components=2)
#     embedding_tsne = tsne.fit_transform(latents)

#     # Create a color map for labels
#     label_color_map = plt.cm.get_cmap('tab10')

#     cmat = confusion_matrix(true_labels, labels)
#     ri, ci = linear_sum_assignment(-cmat)
#     # Plot the embeddings
#     plt.figure(figsize=(8, 8))
#     for label in ri:
#         indices = np.where(labels == ci[label])
#         plt.scatter(embedding_tsne[indices, 0], embedding_tsne[indices, 1], color=label_color_map(label), label=label)

#     # plt.title("s-tne Visualization of Embeddings")
#     plt.legend()
#     # Save the figure as an image file
#     # plt.savefig(file+".png")
#     plt.savefig(file+".pdf")
#     # plt.show()
    
#     # savemat(file+".mat", dict(latents=latents, 
#     #                           embedTSNE=embedding_tsne,
#     #                           predLabels=labels,
#     #                           trueLabels=true_labels))


def write_confumatrix(predicted: np.ndarray, 
                      truelabel: np.ndarray, 
                      clusters: list = None, 
                      filepath: str="",
                      use_orig_clusters:bool=True,
                      eps=0.09):
    import matplotlib.pyplot as plt
    if clusters is None:
        n_cluster = (
                max(predicted.max(), truelabel.max()) + 1
            )  # assume labels are 0-indexed
        clusters = [i for i in range(n_cluster)]
    else:
        n_cluster=len(clusters)
        if not use_orig_clusters:
            clusters = [i for i in range(n_cluster)]
    count_matrix = np.zeros((n_cluster, n_cluster), dtype=np.int64)
    for i in range(predicted.size):
        count_matrix[predicted[i], truelabel[i]] += 1
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]  # TODO numpify
    confusion = confusion_matrix(truelabel, predicted_reassigned)
    normalised_confusion = (
        confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    )

    # Create a mask to hide non-diagonal elements
    annot = np.zeros_like(normalised_confusion).astype(str)
    # np.fill_diagonal(annot, np.round(normalised_confusion.diagonal(), 2))  # 对角线上填入数值
    # annot[annot == '0.0'] = ''  # 非对角线上设置为空字符串
    for i in range(n_cluster):
        for j in range(n_cluster):
            annot[i,j]=str(np.round(normalised_confusion[i,j], 2)) if normalised_confusion[i,j]>eps else ''

    # os.makedirs(filepath,exist_ok=True)
    confusion_id = uuid.uuid4().hex if filepath is None else filepath.split(os.sep)[-1]
    fig = plt.figure()  # 创建新的 Figure 对象
    sns.heatmap(normalised_confusion, 
                xticklabels=clusters,
                yticklabels=clusters,
                cmap='Blues',
                annot=annot,  # Display the values on the heatmap
                fmt="",  # Format the values to 2 decimal places
                # vmin=0.5,  # Set the minimum value for the color scale
                ax=fig.gca()
                )
    fig.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=600) 
    # fig.savefig(f"{filepath}/confusion_{confusion_id}.png" )
    plt.close(fig)  # 关闭 Figure 对象
    savemat(f"{filepath}.mat", dict(CMat=confusion, 
                                    nCMat=normalised_confusion,
                                    clusters=clusters,
                                    labels=truelabel, 
                                    pred_labels=predicted))
    return confusion_id

def visualize_embeddings(latents, labels, true_labels, path=""):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    # Perform s-tne dimensionality reduction
    if isinstance(latents, torch.Tensor):
        latents=latents.detach().cpu().numpy()
    tsne = TSNE(n_components=2)
    embedding_tsne = tsne.fit_transform(latents)

    # Create a color map for labels
    label_color_map = plt.cm.get_cmap('tab10')

    cmat = confusion_matrix(true_labels, labels)
    ri, ci = linear_sum_assignment(-cmat)
    reordered_labels=[]
    reordered_embeddings=[]
    reordered_latents=[]
    # Plot the embeddings
    plt.figure(figsize=(8, 8))
    for label in ri:
        indices = np.where(labels == ci[label])
        reordered_labels.extend(list(labels[indices]))
        reordered_latents.extend(list(latents[indices,...]))
        reordered_embeddings.extend(list(embedding_tsne[indices,:]))
        plt.scatter(embedding_tsne[indices, 0], embedding_tsne[indices, 1], color=label_color_map(label), label=label)

    reordered_labels = np.vstack(reordered_labels)
    reordered_latents = np.vstack(reordered_latents)
    reordered_embeddings = np.vstack(reordered_embeddings)
    # plt.title("S-TNE Visualization of Embeddings")
    plt.legend()
    # Save the figure as an image file
    # plt.savefig(path+".png", bbox_inches='tight')
    plt.savefig(path+".pdf", bbox_inches='tight', dpi=600)
    # plt.show()
    plt.close()
    savemat(path+".mat", dict(embed=reordered_latents,
                              embedTSNE=reordered_embeddings,
                              labels=true_labels,
                              pred_labels=reordered_labels))
    # savemat(path+".mat", dict(embed=latents,
    #                           embedTSNE=embedding_tsne,
    #                           labels=true_labels,
    #                           pred_labels=labels))

def save_clustered_image(images, truelabels, predicts, logdir:str="", max_index:int=10):
    # def ordered_with_cluster(inputs, preds, labels):
    #     inputs = inputs.detach()
    #     preds = preds.detach().cpu().numpy()  
    #     labels= labels.detach().cpu().numpy() 
        
    #     real_clusters = set(list(labels))     
         
    #     outputs = []
    #     ordered_labels=[]
    #     max_cluster=0
    #     for label in real_clusters:
    #         cur_index = np.where(preds==label)[0]
    #         if len(cur_index)>0:
    #             cur_label = pd.value_counts(labels[cur_index]).index[0]
    #             outputs.append(inputs[cur_index])
    #             ordered_labels.append(cur_label)
    #             max_cluster=max(max_cluster,len(cur_index))

    #     ordered_matrix=[]
    #     ordered_labels = torch.from_numpy(np.stack(tuple(ordered_labels))).sort()[1]
    #     for order in ordered_labels:
    #         row = outputs[order]
    #         if len(row)<max_cluster:          
    #             ordered_matrix.append(torch.cat([row,torch.zeros_like(row[0]).repeat(max_cluster-len(row),1,1,1)],0))
    #         else:
    #             ordered_matrix.append(row)

    #     del outputs, ordered_labels
    #     return torch.cat(tuple(ordered_matrix),0), max_cluster
    
    # def set_label_batch_balanced(data, batch_size: int):
    #     labels=np.array(data.tensors[1])        
    #     re_arranged_indices = []
    #     min_num_of_labels = len(labels)
    #     classes = np.unique(labels)
    #     # find sample-indices of each label and the minimum length 
    #     for label in classes:
    #         cur_label_indices = np.where(labels==label)[0]
    #         re_arranged_indices.append(np.stack(tuple(cur_label_indices),0))
    #         if len(cur_label_indices)<min_num_of_labels:
    #             min_num_of_labels = len(cur_label_indices)
        
    #     # the number of samples for each label in a mini-batch
    #     samples_in_batch = batch_size//len(classes) #100/10=10
    #     min_num_of_labels = (min_num_of_labels//samples_in_batch)*samples_in_batch
    #     # cut samples for all labels to the same length (the minimum length ) 
    #     re_arranged_indices = [indices[:min_num_of_labels].reshape(-1,samples_in_batch) for indices in re_arranged_indices]
    #     data = Subset(data, np.stack(re_arranged_indices).transpose(1,0,2).reshape(-1))
    #     return data
    
    if not logdir=="":
        os.makedirs(logdir,exist_ok=True)
    if images.shape[-1]>32:
        transform=transforms.Resize([32,32])
        images=transform(images)
    dataset=TensorDataset(images, torch.from_numpy(truelabels), torch.from_numpy(predicts))
    dataset=set_label_batch_balanced(dataset, 100)
    dataloader = DataLoader(dataset, 100, False, drop_last=True)
    for index, (image, label, cluster) in enumerate(dataloader):
        if index==max_index:
            break
        image_matrix, max_cluster = ordered_with_cluster(image, cluster, label)
        ## save samples for sorted clusters of the current batch as a iamge-matrix
        vutils.save_image(image_matrix, os.path.join(logdir, f"clustered_batch_{index}.png"), normalize=True, nrow=max_cluster)
    
def earlystop(check_value, patience:int=10, eps:float=1.e-3, small_good:bool=True):
    if not hasattr(earlystop, "count") or ( 
        hasattr(earlystop, "errors") and len(earlystop.errors)!=patience+1):
        earlystop.count = 0
    if earlystop.count == 0:
        earlystop.errors=torch.tensor(1).to(check_value.dtype).repeat(patience+1)
        earlystop.check_value=torch.tensor(0).to(check_value.dtype)
    
    earlystop.count += 1
    earlystop.errors[1:]=earlystop.errors[:-1].clone()
    earlystop.errors[0] =(torch.tensor(check_value.item())-earlystop.check_value)

    earlystop.varying_slow=(earlystop.errors.abs()<eps*earlystop.check_value.abs()).all().item()
    earlystop.no_improving=(earlystop.errors<=0).all().item()
    flag = (earlystop.no_improving or earlystop.varying_slow) if earlystop.count>patience else False
    earlystop.check_value=torch.tensor(check_value.item())
    return flag

def get_time():
    timestamp = time.time()
    localtime = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
    return formatted_time

def manual_seed_all(seed: Optional[int] = None, workers: bool = False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the random seed for numpy
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

def plot_confidence_ellipse(data: np.ndarray, file_path: str):
    import matplotlib.pyplot as plt
    import uuid
    from matplotlib.patches import Ellipse

    # 计算数据的均值和协方差矩阵
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # 计算椭圆的参数
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    width, height = 2 * np.sqrt(eigvals) * 2

    # 绘制数据的散点图和置信椭圆
    fig, ax = plt.subplots()
    # ax.scatter(data[:,0], data[:,1], s=3, alpha=0.5)
    # ax.axis('equal')
    ell =Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', lw=2, facecolor='none')
    ell.set_clip_box(ax.bbox)
    ax.add_artist(ell)
    id = uuid.uuid4().hex
    plt.show()
    plt.savefig(os.path.join(file_path, f"{id}.png"))
    plt.savefig(os.path.join(file_path, f"{id}.fig"))
    plt.savefig(os.path.join(file_path, f"{id}.pdf"))
    plt.close(fig)  # 关闭 Figure 对象

def plot_hot(data: np.ndarray, file_name: str, cmap:str="Greys"):
    import matplotlib.pyplot as plt
    # 绘制热力图
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    
    plt.show()
    plt.savefig(f"{file_name}.pdf")
    plt.close(fig)  # 关闭 Figure 对象

def format_mean(data, latex, precision:int=2):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = np.mean(list(data))
    err = np.std(list(data))
    if latex:
        return f"{mean:.{precision}f}\pm{err:.{precision}f}"
    else:
        return f"{mean:.{precision}f}+-{err:.{precision}f}"
    
def print_row(row, colwidth=14, latex=False):    
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = " "*3
        end_ = " "

    def format_val(x):
        cur_colwidth = max(len(x), colwidth)
        if np.issubdtype(type(x), np.floating):
            # x = "{:.10f}".format(x) if colwidth>10 else "{:.6f}".format(x)
            x = f"{round(x, cur_colwidth-2)}"
        return str(x).ljust(cur_colwidth)[:cur_colwidth]
    string=sep.join([format_val(x) for x in row]) + end_
    print(string)
    return string

def print_metrics(metrics, colwidth:int=14, latex:bool=True, filename:str=""):
    suffix = ('.tex' if latex else'.txt')
    sys.stdout = Tee(filename+ (suffix if suffix not in filename else ''), "w")
    df = pd.DataFrame(metrics)
    print_row(list(metrics.keys()), colwidth, latex)
    for i_cv in range(len(df)):
        print_row([df.iloc[i_cv][0]]+[f"{val:.2f}" for val in df.iloc[i_cv][1:]], 
                  colwidth,latex)

    row_str=print_row(["mean"]+[format_mean(df[key],latex) for key in list(metrics.keys())[1:]], 
                      colwidth,latex)
    sys.stdout = sys.__stdout__
    return row_str[colwidth+3:]

class Tee():
    def __init__(self, fname, mode="a"):
        super(Tee, self).__init__()
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.stdout = sys.__stdout__
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class myEventLoader(Dataset):
    # to transform events file to mat file
    # requirement: tensorboard,  get_file() 
    # eventloader = EventLoader(path):  #define a EventLoader with the given path including some *.events files
    # files = eventloader.events_to_mat(file_num=0): # then call this function to transform a file of given file_nums,
    #   here file_num can be int scale or int list 
    
    def __init__(self, file_path) -> None:
        self.file_path = get_file(file_path, part_name="events.out.tfevents.")[0] 
        # Numbers = [str(num) for num in range(10)]
        # self.file_path = [file for file in file_path if os.path.splitext(file)[-1][-1] in Numbers]
        if self.file_path==[]:
            warning("There're no events files in the given path")     
        
        self.Key = []
        self.Data= []
        self.EA  = None
        
    def readFile(self, **kwargs):
        from tensorboard.backend.event_processing import event_accumulator
        file_path = self.file_path[0] if kwargs.get('file_path') is None else kwargs['file_path']
        ea = event_accumulator.EventAccumulator(file_path)
        ea.Reload()
        self.EA = ea.scalars
        self.Key = [key for key in ea.scalars.Keys()]
        self.Data = {key: ea.scalars.Items(key) for key in self.Key}

    def get_item(self, index_key=-1, index_item=-1) -> Any:
        # index_key:  key   index to transform, self.Key[index_key]
            # default sets to -1 if transforming all the Keys in data
        # index_item: items index to transform, self.data[key][index_item]
            # default sets to -1 if transforming all the items of given Key in data
        if index_key!=-1:
            index_key = [index_key] if isinstance(index_key, int) else index_key        
            Key  = [self.Key[key] for key in index_key if ( key <len(self.Key)  ) ]   
        else:
            Key  = self.Key
            
        result = {}
        for key in Key:
            if index_item != -1:
                index_out = [index_item] if isinstance(index_item, int) else index_item
            else:
                len_item = len(self.Data[key])
                index_out = [index for index in range(len_item)]
            items = [self.Data[key][index].value for index in index_out]
            if 'lr-' in key: # don't log lr
                key='lr_'+key[3:]
            if len(items)>1:    
                result[key] = list(set(np.array(items).squeeze().astype(np.int32).tolist())) if key=='epoch' else items
            elif len(items)==1:  
                result[key] = list(int([0])) if key=='epoch' else items
            else:
                result[key] = []
        return result

    def events_to_mat(self, file_num=-1, write_file=True):    
        # call this function to convert the events files of given number in path to *.mat
        from scipy.io import savemat
        if file_num==-1:
            file_list = [num for num in range(len(self.file_path))]
        else:
            file_list = [file_num] if isinstance(file_num, int) else file_num
        mat_files = []
        for num in file_list:     
            mat_file = self.file_path[num]+".mat" 
            if os.path.exists(mat_file):
                mat_files.append(mat_file)
            else:
                if num>=len(self.file_path): 
                    continue    
                self.readFile(file_path=self.file_path[num])
                results = self.get_item(index_item = -1, index_key = -1)
                mat_files.append(mat_file)
                if results=={}:
                    continue
                else:
                    if write_file:
                        savemat(mat_files[num], results)
        
        return mat_files
    
    def events_to_yaml(self, file_num=-1):   
        from scipy.io import loadmat
        import yaml
        if file_num==-1:
            file_list = [num for num in range(len(self.file_path))]
        else:
            file_list = [file_num] if isinstance(file_num, int) else file_num
        for num in file_list:     
            eventfile = self.file_path[num]    
            try:
                data=loadmat(eventfile+".mat")
                if not data=={}:
                    with open(eventfile+".yaml", 'w') as file:
                        yaml.dump(data, file) 
            except:
                pass

class myTensorboard(object):
    def __init__(self, log_dir, run_version, delay=20):
        import multiprocessing
        active_processes, active_ports, exist = self.get_active(log_dir)

        if exist:      
            tb_process=active_processes
            start=ready=time.time()
        else:
            port = (max(active_ports)+1) if (6001+run_version) in active_ports else (6001+run_version)       
            tb_started = multiprocessing.Event()
            tb_process = multiprocessing.Process(target=self.activate, args=(log_dir, port, tb_started))
            tb_process.port = port
        self.process=tb_process
        self.started=tb_started
        self.port = tb_process.port
        self.pid = tb_process.pid
    
    def start(self,):
        # time.sleep(5)
        self.process.start()        
        # 等待TensorBoard进程启动完成
        self.started.wait()   
        # 等待TensorBoard进程结束
        self.process.join()

    @staticmethod
    def activate(log_dir, port, event):
        import subprocess, webbrowser
        # 构造TensorBoard命令行参数
        cmd = f"tensorboard --logdir={log_dir} --port={port}"
        # 启动TensorBoard进程
        tb_process = subprocess.Popen(cmd, shell=True)      
        # 构造TensorBoard URL地址
        tb_url = f"http://localhost:{port}"
        # 自动打开TensorBoard页面
        webbrowser.open(tb_url)
        
        # 设置事件，通知主进程TensorBoard进程已启动完成
        event.set()
        return tb_process

    @staticmethod
    def get_active(log_dir):
        # 遍历进程列表，查找并关闭活动的TensorBoard进程
        import psutil
        # 获取tensorboard进程然后关闭进程
        Process, Port , exist = [], [], False
        for process in psutil.process_iter():
            try:
                if "tensorboard" in process.name().lower():
                    try:
                        # 获取TensorBoard进程的端口号   
                        port = int(process.cmdline()[process.cmdline().index("--port")+1])    
                        # 获取TensorBoard进程的logdir   
                        logdir = process.cmdline()[process.cmdline().index("--logdir")+1] 
                        if logdir == log_dir:
                            Process, Port, exist = process, port, True
                            Process.port = port
                            break
                        else:
                            Process.append(process)
                            Port.append(port)
                    except:
                        pass

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return Process, Port, exist

    @staticmethod
    def close_id(tb_pid):
        import subprocess
        # 在需要关闭TensorBoard进程时，使用PID来终止进程
        subprocess.call(["kill", str(tb_pid)])

    @staticmethod
    def close_existing():
        # 遍历进程列表，查找并关闭活动的TensorBoard进程
        import psutil
        # 获取tensorboard进程然后关闭进程
        for process in psutil.process_iter():
            try:
                if "tensorboard" in process.name().lower():
                    # 获取TensorBoard进程的端口号          
                    for name in process.cmdline():
                        pos = name.find("port")
                        if pos!=-1:
                            port = int(name[pos+5:])
                    # 终止TensorBoard进程
                    process.terminate()
                    print(f"TensorBoard process with PID-{process.pid} @port-{port} has been terminated.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
