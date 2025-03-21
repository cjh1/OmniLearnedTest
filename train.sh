# load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train.py  -o ./ --save_tag qg --dataset qg --use_pid"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
