import json
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

from network import PET2
from dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from utils import print_metrics, sum_reduce, is_master_node, ddp_setup,get_param_groups
import torch.nn.functional as F  # Keep this for function calls

    
def train_step(model, dataloader, cost, optimizer,scheduler, epoch, device):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):       
        optimizer.zero_grad()  # Zero the gradients
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        model_kwargs['time'] = torch.rand(size=(X.shape[0],)).to(device)*0.0

        outputs = model(X, **model_kwargs)
        loss = cost(outputs.squeeze(), y)

        loss.backward()  # Backward pas
        optimizer.step()  # Update parameters
        
        running_loss += loss.item()
        scheduler.step()

    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch
    return distributed_loss

def test_step(model, dataloader, cost, epoch, device):
    model.eval()
    running_loss = 0.0
    ys = []
    preds = []
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        model_kwargs['time'] = torch.zeros(size=(X.shape[0],),dtype=torch.float).to(device)
        #model_kwargs = {key: batch[key].to(device) if batch[key] is not None else None for key in ["cond", "pid","add_info"] if key in batch}
        with torch.no_grad():
            outputs = model(X, **model_kwargs)            
        
        loss = cost(outputs.squeeze(), y)
        
        preds.append(outputs.squeeze())
        ys.append(y)
        running_loss += loss.item()


    distributed_batch = sum_reduce(len(dataloader), device=device).item()
    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch

    local_preds = torch.cat(preds)
    local_ys = torch.cat(ys)

    # Allocate lists to store gathered tensors
    world_size = dist.get_world_size()
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_ys = [torch.zeros_like(local_ys) for _ in range(world_size)]

    
    # Perform all_gather
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_ys, local_ys)

    # Concatenate results from all GPUs
    gathered_preds = torch.cat(gathered_preds)
    gathered_ys = torch.cat(gathered_ys)
    if is_master_node():
        print_metrics(gathered_preds,gathered_ys)

    return distributed_loss

def train_model(model,
                train_loader,
                test_loader,
                optimizer,
                num_epochs=1,
                device='cpu',
                patience=10,
                loss = nn.CrossEntropyLoss(),
                output_dir="",
                save_tag=""):

    

    checkpoint_name = f"best_model_{save_tag}.pt"
    
    losses = {"train_loss": [],
              "val_loss": [],}
   
    tracker = {"bestValLoss": np.inf,
               "bestEpoch": 0}

    lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(len(train_loader) * num_epochs))
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_losses = train_step(model, train_loader, loss,
                                  optimizer, lr_scheduler,
                                  epoch, device)
        val_losses = test_step(model, test_loader, loss, epoch, device)
        
        losses["train_loss"].append(train_losses)
        losses["val_loss"].append(val_losses)

        if is_master_node():
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4f}, Val Loss: {losses["val_loss"][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}')

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
            tracker["bestValLoss"] = losses["val_loss"][-1]
            tracker["bestEpoch"] = epoch
            
            if is_master_node():
                torch.save(model.module.state_dict(), f"{output_dir}/{checkpoint_name}")
                    
        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break
        
    if is_master_node():
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")                
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))

def main(args=None):
    local_rank,rank = ddp_setup()

    # set up model
    model = PET2(input_dim=4,
                 hidden_size=args.base_dim,
                 num_transformers = args.num_transf,
                 num_heads = args.num_head,
                 attn_drop = args.attn_drop,
                 mlp_drop = args.mlp_drop,
                 mlp_ratio=args.mlp_ratio,
                 num_tokens = args.num_tokens,
                 K = args.K,
                 cut=0.3,                 
                 conditional = True,
                 use_time=True,
                 pid = args.use_pid,
                 num_classes = args.num_classes)

    if rank==0:
        print('**** Setup ****')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')

    param_groups = get_param_groups(model,args.wd)
    model = DDP(model.to(local_rank), device_ids=[local_rank])
    optimizer = Lion(param_groups, lr=args.lr,betas=(args.b1,args.b2))

    if rank==0:
        d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {d}")

    # load in train data
    train_loader = load_data(args.dataset,dataset_type='train',
                             use_pid = args.use_pid,
                             use_add = args.use_add,
                             path=args.path,
                             batch = args.batch)
    test_loader = load_data(args.dataset,dataset_type='test',
                            use_pid = args.use_pid,
                            use_add = args.use_add,
                            path=args.path,
                            batch = args.batch)

    train_model(model, 
                train_loader, 
                test_loader, 
                optimizer,
                num_epochs = args.epoch,
                device=local_rank, 
                output_dir=args.outdir, 
                save_tag=args.save_tag,
                )


    #destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()

    #General Options
    parser.add_argument("-o","--output_dir",dest="outdir",default="",help="Directory to output best model",)
    parser.add_argument("--save_tag",dest="save_tag",default="",help="Extra tag for checkpoint model",)
    parser.add_argument("--dataset",default="top",help="Dataset to load")
    parser.add_argument("--path",default="/pscratch/sd/v/vmikuni/PET/",help="Dataset path")

    #Model Options
    parser.add_argument("--use_pid", action='store_true', default=False, help='Use particle ID for training')
    parser.add_argument("--use_add", action='store_true', default=False, help='Use additional features beyond kinematic information')
    parser.add_argument("--num_classes",default=2,type = int,help="Number of classes in the classification task")
    parser.add_argument("--mode",default="classifier",help="Task to run: classifier, diffusion, pretrain")


    #Training options
    parser.add_argument("--batch",default=64,type = int,help="Batch size",)    
    parser.add_argument("--epoch",default=20,type = int,help="Number of epochs")

    #Optimizer
    parser.add_argument("--b1",default=0.95, type = float,help="Lion b1")
    parser.add_argument("--b2",default=0.98,type = float,help="Lion b2")
    parser.add_argument("--lr",default=5e-4,type = float,help="Learning rate")
    parser.add_argument("--wd",default=0.3,type = float,help="Weight decay")

    #Model
    parser.add_argument("--num_transf",default=6,type = int,help="Number of transformer blocks")
    parser.add_argument("--num_tokens",default=4,type = int,help="Number of trainable tokens")
    parser.add_argument("--num_head",default=8,type = int,help="Number of transformer heads")
    parser.add_argument("--K",default=15,type = int,help="Number of nearest neighbors")
    parser.add_argument("--base_dim",default=64,type = int,help="Base value for dimensions")
    parser.add_argument("--mlp_ratio",default=2,type = int,help="Multiplier for MLP layers")
    parser.add_argument("--attn_drop",default=0.1,type = float,help="Dropout for attention layers")
    parser.add_argument("--mlp_drop",default=0.1,type = float,help="Dropout for mlp layers")

    args = parser.parse_args()
    main(args)
