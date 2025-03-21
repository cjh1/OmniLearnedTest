from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

from network import PET2
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, get_rank


def print_metrics(y_pred, y, thresholds=[0.3,0.5]):
    y = y.cpu().detach().numpy()[:,1]
    y_pred = nn.functional.softmax(y_pred,-1).cpu().detach().numpy()[:,1]
    print("AUC: {}".format(metrics.roc_auc_score(y, y_pred)))
    print('Acc: {}'.format(metrics.accuracy_score(y,y_pred>0.5)))
        
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)

    for threshold in thresholds:
        bineff = np.argmax(tpr>threshold)
        print('effS at {} 1.0/effB = {}'.format(tpr[bineff],1.0/fpr[bineff]))


def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_param_groups(model,wd):
    no_decay = []
    decay = []
        
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in model.no_weight_decay()):
            no_decay.append(param)  # Exclude from weight decay
        else:
            decay.append(param)  # Apply weight decay

    param_groups = [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

    return param_groups


        
def is_master_node():
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    else:
        return True

def ddp_setup():
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2900"
        os.environ["RANK"] = "0"
        init_process_group(backend="nccl", rank=0, world_size=1)
        rank = local_rank = 0
    else:
        init_process_group(backend="nccl", 
                           init_method='env://')
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()
    
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank, rank

        
