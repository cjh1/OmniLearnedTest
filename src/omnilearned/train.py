import json
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from omnilearned.utils import print_metrics, sum_reduce, is_master_node, ddp_setup,get_param_groups, CLIPLoss
import torch.nn.functional as F  # Keep this for function calls
import time, sys, os

def train_step(
        model,
        dataloader,
        class_cost,
        gen_cost,
        optimizer,
        scheduler,
        epoch,
        device,
        clip_loss = CLIPLoss(),
        use_clip = False,
        iterations_per_epoch = -1,
):
    model.train()

    logs_buff = torch.zeros((5), dtype=torch.float32, device=device)
    logs = {}
    logs['loss'] = logs_buff[0].view(-1)
    logs['loss_class'] = logs_buff[1].view(-1)
    logs['loss_gen'] = logs_buff[2].view(-1)
    logs['loss_perturb'] = logs_buff[3].view(-1)
    logs['loss_clip'] = logs_buff[4].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

    #for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }


        y_pred,y_perturb,z_pred, v, x_body, z_body = model(X,y, **model_kwargs)

        loss = 0
        if y_pred is not None:
            loss_class = class_cost(y_pred.squeeze(), y)
            loss = loss + loss_class
            logs['loss_class'] += loss_class.detach()
        if z_pred is not None:
            loss_gen = gen_cost(v,z_pred)
            loss = loss + loss_gen
            logs['loss_gen'] += loss_gen.detach()
        if y_perturb is not None:
            loss_perturb = class_cost(y_perturb.squeeze(), y)
            loss = loss + loss_perturb
            logs['loss_perturb'] += loss_perturb.detach()
        if use_clip and z_body is not None and x_body is not None:
            loss_clip = clip_loss(x_body.view(X.shape[0],-1),z_body.view(X.shape[0],-1))
            loss = loss + loss_clip
            logs['loss_clip'] += loss_clip.detach()

        logs['loss'] += loss.detach()
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        scheduler.step()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key]/dist.get_world_size()/len(dataloader))

    return logs

def test_step(model,
              dataloader,
              class_cost,
              gen_cost,
              epoch,
              device,
              clip_loss = CLIPLoss(),
              use_clip=False,
              iterations_per_epoch = -1,):
    model.eval()

    logs_buff = torch.zeros((5), dtype=torch.float32, device=device)
    logs = {}
    logs['loss'] = logs_buff[0].view(-1)
    logs['loss_class'] = logs_buff[1].view(-1)
    logs['loss_gen'] = logs_buff[2].view(-1)
    logs['loss_perturb'] = logs_buff[3].view(-1)
    logs['loss_clip'] = logs_buff[4].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

    #for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        with torch.no_grad():
            y_pred,y_perturb,z_pred, v, x_body, z_body = model(X,y, **model_kwargs)

        loss = 0
        if y_pred is not None:
            loss_class = class_cost(y_pred.squeeze(), y)
            loss = loss + loss_class
            logs['loss_class'] += loss_class.detach()
        if z_pred is not None:
            loss_gen = gen_cost(v,z_pred)
            loss = loss + loss_gen
            logs['loss_gen'] += loss_gen.detach()
        if y_perturb is not None:
            loss_perturb = class_cost(y_perturb.squeeze(), y)
            loss = loss + loss_perturb
            logs['loss_perturb'] += loss_perturb.detach()

        if use_clip and z_body is not None and x_body is not None:
            loss_clip = clip_loss(x_body.view(X.shape[0],-1),z_body.view(X.shape[0],-1))
            loss = loss + loss_clip
            logs['loss_clip'] += loss_clip.detach()

        logs['loss'] += loss.detach()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key]/dist.get_world_size()/len(dataloader))

    return logs

def train_model(model,
                train_loader,
                test_loader,
                optimizer,
                num_epochs=1,
                device='cpu',
                patience=100,
                loss_class = nn.CrossEntropyLoss(),
                loss_gen = nn.L1Loss(),
                use_clip=True,
                output_dir="",
                save_tag="",
                iterations_per_epoch = -1):



    checkpoint_name = f"best_model_{save_tag}.pt"

    losses = {"train_loss": [],
              "val_loss": [],}

    tracker = {"bestValLoss": np.inf,
               "bestEpoch": 0}

    lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(len(train_loader) * num_epochs))
    epoch_init = 0
    if os.path.isfile(os.path.join(output_dir,checkpoint_name)):
        if is_master_node(): print(f"Loading checkpoint from {os.path.join(output_dir,checkpoint_name)}")
        epoch_init, tracker["bestValLoss"] = restore_checkpoint(model,optimizer,lr_scheduler,
                                                                output_dir,checkpoint_name,device)


    for epoch in range(int(epoch_init),num_epochs):
        train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(model,train_loader,
                                loss_class,loss_gen,
                                optimizer,lr_scheduler,
                                epoch,device,
                                use_clip = use_clip,
                                iterations_per_epoch = iterations_per_epoch)
        val_logs = test_step(model, test_loader,
                             loss_class, loss_gen,
                             epoch, device,use_clip=use_clip,
                             iterations_per_epoch = iterations_per_epoch)

        losses["train_loss"].append(train_logs['loss'])
        losses["val_loss"].append(val_logs['loss'])

        if is_master_node():
            print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4f}, Val Loss: {losses["val_loss"][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}')
            print(f'Class Loss: {train_logs["loss_class"]:.4f}, Class Val Loss: {val_logs["loss_class"]:.4f}')
            print(f'Gen Loss: {train_logs["loss_gen"]:.4f}, Gen Val Loss: {val_logs["loss_gen"]:.4f}')
            print(f'Class Perturb Loss: {train_logs["loss_perturb"]:.4f}, Class Val Perturb Loss: {val_logs["loss_perturb"]:.4f}')
            print(f'CLIP loss: {train_logs["loss_clip"]:.4f}, CLIP Val Loss: {val_logs["loss_clip"]:.4f}')
            print('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))



        if is_master_node() and  losses["val_loss"][-1] < tracker["bestValLoss"]:
            print("replacing best checkpoint ...")
            tracker["bestValLoss"] = losses["val_loss"][-1]
            save_checkpoint(model,epoch+1,optimizer,
                            losses["val_loss"][-1],
                            lr_scheduler,output_dir,
                            checkpoint_name)

        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break

    if is_master_node():
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))


def save_checkpoint(model,
                    epoch,
                    optimizer,
                    loss,
                    lr_scheduler,
                    checkpoint_dir,
                    checkpoint_name
                    ):
    save_dict = {
        'body': model.module.body.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':epoch,
        'loss':loss,
        'sched': lr_scheduler.state_dict(),
    }

    if model.module.classifier is not None:
        save_dict['classifier_head'] = model.module.classifier.state_dict()

    if model.module.generator is not None:
        save_dict['generator_head'] = model.module.generator.state_dict()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(save_dict, os.path.join(checkpoint_dir,checkpoint_name))
    print(f"Epoch {epoch} | Training checkpoint saved at {os.path.join(checkpoint_dir,checkpoint_name)}")

def restore_checkpoint(model,
                       optimizer,
                       lr_scheduler,
                       checkpoint_dir,
                       checkpoint_name,
                       device):
    checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_name),
                            map_location='cuda:{}'.format(device))
    model.module.body.load_state_dict(checkpoint["body"],strict=False)

    if model.module.classifier is not None:
        model.module.classifier.load_state_dict(checkpoint["classifier_head"],strict=False)

    if model.module.generator is not None:
        model.module.generator.load_state_dict(checkpoint["generator_head"], strict=False)


    startEpoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except Exception as e:
        print("Optimizer cannot be loaded back, skipping...")
    lr_scheduler.load_state_dict(checkpoint['sched'])
    return startEpoch, best_loss

def run(outdir: str = "",
        save_tag: str = "",
        dataset: str = "top",
        path: str = "/pscratch/sd/v/vmikuni/PET/datasets",
        use_pid: bool = False,
        use_add: bool = False,
        use_clip: bool = False,
        num_classes: int = 2,
        mode: str = "classifier",
        batch: int = 64,
        iterations: int = -1,
        epoch: int = 15,
        b1: float = 0.95,
        b2: float = 0.98,
        lr: float = 5e-4,
        wd: float = 0.3,
        num_transf: int = 6,
        num_tokens: int = 4,
        num_head: int = 8,
        K: int = 15,
        radius: int = 0.4,
        base_dim: int = 64,
        mlp_ratio: int = 2,
        attn_drop: float = 0.1,
        mlp_drop: float = 0.1,
        feature_drop: float = 0.0):

    local_rank,rank = ddp_setup()

    # set up model
    model = PET2(input_dim=4,
                 hidden_size=base_dim,
                 num_transformers = num_transf,
                 num_heads = num_head,
                 attn_drop = attn_drop,
                 mlp_drop = mlp_drop,
                 mlp_ratio=mlp_ratio,
                 feature_drop = feature_drop,
                 num_tokens = num_tokens,
                 K = K,
                 cut=radius,
                 conditional = True,
                 use_time=True,
                 pid = use_pid,
                 num_classes = num_classes,
                 mode = mode)

    if rank==0:
        print('**** Setup ****')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')

    param_groups = get_param_groups(model,wd)
    model = DDP(model.to(local_rank), device_ids=[local_rank],
                #find_unused_parameters=True
                )
    optimizer = Lion(param_groups, lr=lr,betas=(b1, b2))

    if rank==0:
        d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {d}")

    # load in train data
    train_loader = load_data(dataset,dataset_type='train',
                             use_pid = use_pid,
                             use_add = use_add,
                             path=path,
                             batch = batch)
    if rank==0:
        print('**** Setup ****')
        print(f'Train dataset len: {len(train_loader)}')
        print('************')

    test_loader = load_data(dataset,dataset_type='test',
                            use_pid = use_pid,
                            use_add = use_add,
                            path=path,
                            batch = batch)

    train_model(model,
                train_loader,
                test_loader,
                optimizer,
                num_epochs = epoch,
                device=local_rank,
                output_dir=outdir,
                save_tag=save_tag,
                use_clip = use_clip,
                iterations_per_epoch = iterations
                )


    #destroy_process_group()

