python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
    --finetune models/origin_teacher/S24_224.pth \
    --distillation-type none \
    --resume models/student/cait-s24_deit-tiny/checkpoint.pth \
    --epochs 60 \
    --output_dir models/finetune_teacher_on_imagenet/MCAMCait_s24_224 \
    --data-set VOC12MS \
    --data-path datasets/voc12/VOCdevkit/VOC2012 \
    --model MCAMCait_s24_224 \
    --teacher-model MCAMCait_s24_224 \
    --teacher-path models/origin_teacher/S24_224.pth \
    --model-ema \
    --enable-mixup \
    --enable-smoothing \
    --distillation-alpha 1 \
    --distillation-beta 1 \
    --distillation-gamma 1 \
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
    --attention-type fused \
    --multilabel \
    --gen-attention-maps \
    --visualize-cls-attn \
    --img-list datasets/voc12 \
    --label-file-path datasets/voc12/cls_labels.npy \
    --scales 1.0 \
    --attention-dir cam_results/test/attn-patchrefine \
    --cam-npy-dir cam_results/test/attn-patchrefine-npy \
    --out-crf cam_results/test/attn-patchrefine-npy-crf \
    2>&1 | tee train.log





