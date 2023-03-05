# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#     --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
#     --master_port=$ARNOLD_WORKER_0_PORT  
python3 main.py \
    --finetune models/finetune_student_on_imagenet/MCAMDeit_tiny_patch16_224/checkpoint_best.pth \
    --distillation-type soft \
    --epochs 100 \
    --output_dir models/cait-xxs24_deit-ti_on_voc12/all \
    --data-set VOC12 \
    --data-path datasets/voc12/VOCdevkit/VOC2012 \
    --model MCAMDeit_tiny_patch16_224 \
    --teacher-model MCAMCait_xxs24_224 \
    --teacher-path models/finetune_pre_teacher_on_voc12/MCAMCait_xxs24_224/checkpoint_best.pth \
    --model-ema \
    --enable-mixup \
    --multilabel \
    --distillation-alpha 1 \
    --distillation-beta 1 \
    --distillation-gamma 1 \
    --distillation-tau 1 \
    --w-patch 4 \
    --w-sample 0.1 \
    --w-rand 0.2 \
    --K 192 \
    --s-id 0 1 2 3 8 9 10 11 \
    --t-id 0 1 2 3 20 21 22 23 \
    --drop-path 0 \
    --patch-attn-refine \
    --patch-size 16 \
    --n-layers-t 2 \
    --n-layers 3 \
    --attention-type patchcam \
    --img-list datasets/voc12 \
    --label-file-path datasets/voc12/cls_labels.npy \
    2>&1 | tee logs/cait-xxs24_deit-ti_on_voc12_patchcam.log





