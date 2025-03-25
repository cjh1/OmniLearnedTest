from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from typing import Tuple

from omnilearned.network import PET2
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, get_rank
import torch.nn.functional as F

def print_metrics(y_preds, y, thresholds=[0.3,0.5],background_class=0):

    y_preds_np = F.softmax(y_preds,-1).detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(y_np, y_preds_np if y_preds_np.shape[-1]>2 else np.argmax(y_preds_np,-1), multi_class='ovo')
    print(f"AUC: {auc_ovo:.4f}\n")

    num_classes = y_preds.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y_np == signal_class) | (y_np == background_class)
        y_bin = (y_np[mask] == signal_class).astype(int)
        scores_bin = y_preds_np[mask, signal_class]/(y_preds_np[mask, signal_class] + y_preds_np[mask, background_class])

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr>threshold)
            print('Class {} effS at {} 1.0/effB = {}'.format(signal_class,tpr[bineff],1.0/fpr[bineff]))

class CLIPLoss(nn.Module):
    #From AstroCLIP: https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py#L117
    def get_logits(
            self,
            clean_features: torch.FloatTensor,
            perturbed_features: torch.FloatTensor,
            logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features

        logits_per_clean = logit_scale * clean_features @ perturbed_features.T
        return logits_per_clean, logits_per_clean.T

    def forward(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float = 2.74,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the clean and perturbed features
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels)
            + F.cross_entropy(logits_per_perturbed, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss




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
