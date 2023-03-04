# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
cd .. && git clone -b manifold_cam https://github.com/HandH1998/manifold_distillation.git && \
cd manifold_distillation

pip install -r requirements.txt

mkdir -p datasets/imagenet && cd datasets/imagenet
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/datasets/imagenet/train.tar.gz && \
tar -xzvf train.tar.gz && rm -rf train.tar.gz
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/datasets/imagenet/val.tar.gz && \
tar -xzvf val.tar.gz && rm -rf val.tar.gz

cd .. && cd ..

mkdir -p models/ && cd models && \
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/finetune_teacher_on_imagenet
cd ..
# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
    --master_port=$ARNOLD_WORKER_0_PORT  main.py \
    --resume models/finetune_teacher_on_imagenet/MCAMCait_xxs24_224/checkpoint.pth \
    --distillation-type none \
    --epochs 430 \
    --output_dir models/finetune_teacher_on_imagenet/MCAMCait_xxs24_224 \
    --data-set IMNET \
    --data-path datasets/imagenet \
    --model MCAMCait_xxs24_224 \
    --model-ema \
    --enable-mixup \
    --drop-path 0.05 \
    2>&1 | tee train.log

mv train.log models/finetune_teacher_on_imagenet/MCAMCait_xxs24_224
hdfs dfs -put models/finetune_teacher_on_imagenet/MCAMCait_xxs24_224 \
/home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/finetune_teacher_on_imagenet/





  





