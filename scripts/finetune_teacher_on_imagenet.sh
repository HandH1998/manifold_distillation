# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
    --master_port=$ARNOLD_WORKER_0_PORT  main.py \
    --finetune models/origin_teacher/S24_224.pth \
    --distillation-type none \
    --epochs 60 \
    --output_dir models/finetune_teacher_on_imagenet/MCAMCait_s24_224 \
    --data-set IMNET \
    --data-path datasets/imagenet \
    --model MCAMCait_s24_224 \
    --model-ema \
    --enable-mixup \
    --drop-path 0 \
    2>&1 | tee train.log

  




  





