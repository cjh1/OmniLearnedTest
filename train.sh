# load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train.py  -o ./ --save_tag pretrain --dataset pretrain --use_pid --use_add --num_classes 13 --batch 128 --iterations 1000 --mode pretrain --use_clip --epoch 200"
#cmd="python train.py  -o ./ --save_tag top --dataset top  --num_classes 2"
#cmd="python train.py  -o ./ --save_tag top --dataset top  --num_classes 2 --mode pretrain --use_clip"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
