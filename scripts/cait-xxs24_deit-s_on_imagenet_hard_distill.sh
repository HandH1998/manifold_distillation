# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#     --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
#     --master_port=$ARNOLD_WORKER_0_PORT  
python3 main.py \
    --finetune models/origin_student/deit_small_patch16_224-cd65a155.pth \
    --distillation-type hard \
    --epochs 300 \
    --output_dir models/cait-xxs24_deit-s_on_imagenet_hard_distill \
    --data-set IMNET \
    --data-path datasets/imagenet \
    --model MCAMDeit_small_patch16_224 \
    --teacher-model MCAMCait_xxs24_224 \
    --teacher-path models/finetune_teacher_on_imagenet/MCAMCait_xxs24_224/checkpoint_best.pth \
    --model-ema \
    --enable-mixup \
    --distillation-alpha 0.5 \
    --distillation-beta 0.0 \
    --distillation-gamma 0.0 \
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




