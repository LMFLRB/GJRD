import torch, os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset

def get_father_path(order=1):
    father=os.path.dirname(__file__)
    for i in range(order):
        father=os.path.dirname(father)
    return father

def ordered_with_cluster(inputs, preds, labels):
    inputs = inputs.detach()
    preds = preds.detach().cpu().numpy()  
    labels= labels.detach().cpu().numpy() 
    
    real_clusters = set(list(labels))     
        
    outputs = []
    ordered_labels=[]
    max_cluster=0
    for label in real_clusters:
        cur_index = np.where(preds==label)[0]
        if len(cur_index)>0:
            cur_label = pd.value_counts(labels[cur_index]).index[0]
            outputs.append(inputs[cur_index])
            ordered_labels.append(cur_label)
            max_cluster=max(max_cluster,len(cur_index))

    ordered_matrix=[]
    ordered_labels = torch.from_numpy(np.stack(tuple(ordered_labels))).sort()[1]
    for order in ordered_labels:
        row = outputs[order]
        if len(row)<max_cluster:          
            ordered_matrix.append(torch.cat([row,torch.zeros_like(row[0]).repeat(max_cluster-len(row),1,1,1)],0))
        else:
            ordered_matrix.append(row)

    del outputs, ordered_labels
    return torch.cat(tuple(ordered_matrix),0), max_cluster
    
def set_label_batch_balanced(data, batch_size: int):
    labels=np.array(data.tensors[1])        
    re_arranged_indices = []
    min_num_of_labels = len(labels)
    classes = np.unique(labels)
    # find sample-indices of each label and the minimum length 
    for label in classes:
        cur_label_indices = np.where(labels==label)[0]
        re_arranged_indices.append(np.stack(tuple(cur_label_indices),0))
        if len(cur_label_indices)<min_num_of_labels:
            min_num_of_labels = len(cur_label_indices)
    
    # the number of samples for each label in a mini-batch
    samples_in_batch = batch_size//len(classes) #100/10=10
    min_num_of_labels = (min_num_of_labels//samples_in_batch)*samples_in_batch
    # cut samples for all labels to the same length (the minimum length ) 
    re_arranged_indices = [indices[:min_num_of_labels].reshape(-1,samples_in_batch) for indices in re_arranged_indices]
    data = Subset(data, np.stack(re_arranged_indices).transpose(1,0,2).reshape(-1))
    return data
