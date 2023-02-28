python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --finetune models/origin_teacher/S24_224.pth \
    --distillation-type none \
    --epochs 50 \
    --output_dir models/fine_tune_teacher_on_imagenet \
    --data-set IMNET \
    --data-path datasets/imagenet \
    --model MCAMCait_s24_224 \
    --model-ema \
    --enable-mixup \
    --drop-path 0 
  





