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
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/origin_student
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/finetune_teacher_on_imagenet_400epoch
cd ..
# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \

python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
    --master_port=$ARNOLD_WORKER_0_PORT  main.py \
    --finetune models/origin_student/deit_tiny_patch16_224-a1311bcf.pth \
    --distillation-type soft \
    --epochs 300 \
    --output_dir models/cait-xxs24_deit-ti_on_imagenet \
    --data-set IMNET \
    --data-path datasets/imagenet \
    --model MCAMDeit_tiny_patch16_224 \
    --teacher-model MCAMCait_xxs24_224 \
    --teacher-path models/finetune_teacher_on_imagenet_400epoch/MCAMCait_xxs24_224/checkpoint_best.pth \
    --model-ema \
    --enable-mixup \
    --distillation-alpha 0.5 \
    --distillation-beta 1.0 \
    --distillation-gamma 2.0 \
    --distillation-tau 1 \
    --w-patch 4 \
    --w-sample 0.0 \
    --w-rand 0.2 \
    --K 192 \
    --s-id 0 1 2 3 8 9 10 11 \
    --t-id 0 1 2 3 20 21 22 23 \
    --drop-path 0 \
    --patch-attn-refine \
    --patch-size 16 \
    --n-layers-t 2 \
    --n-layers 3 \
    --attention-type fused

