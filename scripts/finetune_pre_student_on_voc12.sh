# python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
# cd .. && git clone -b manifold_cam https://github.com/HandH1998/manifold_distillation.git && \
# cd manifold_distillation

# pip install -r requirements.txt

# mkdir -p datasets/imagenet && cd datasets/imagenet
# hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/datasets/imagenet/train.tar.gz && \
# tar -xzvf train.tar.gz && rm -rf train.tar.gz
# hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/datasets/imagenet/val.tar.gz && \
# tar -xzvf val.tar.gz && rm -rf val.tar.gz

# cd .. && cd ..

# mkdir -p models/ && cd models && \
# hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/origin_student
# cd ..

# python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU \
#     --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST \
#     --master_port=$ARNOLD_WORKER_0_PORT  
python3 main.py \
    --finetune models/finetune_student_on_imagenet/MCAMDeit_small_patch16_224/checkpoint_best.pth \
    --distillation-type none \
    --epochs 60 \
    --output_dir models/finetune_pre_student_on_voc12/MCAMDeit_small_patch16_224 \
    --data-set VOC12 \
    --data-path datasets/voc12/VOCdevkit/VOC2012 \
    --model MCAMDeit_small_patch16_224 \
    --model-ema \
    --enable-mixup \
    --drop-path 0.1 \
    --img-list datasets/voc12 \
    --label-file-path datasets/voc12/cls_labels.npy \
    --multilabel \
    2>&1 | tee logs/finetune_pre_deit_s_on_voc12_train_60epoch.log

# mv train.log models/finetune_student_on_imagenet/MCAMDeit_small_patch16_224
# hdfs dfs -put models/finetune_student_on_imagenet/MCAMDeit_small_patch16_224/ \
# /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/finetune_student_on_imagenet/



  




  





