# 单GPU或cpu训练
# --curriculum_learning 开启课程学习
# --distill 开启蒸馏
CUDA_VISIBLE_DEVICES=0 python main.py --train \
    --arch efficientnet_b0 --num_classes 3 \
    --criterion bce --weighted_loss --ohm_loss --ghm_loss --threshold_loss \
    --opt adam \
    --epoches 65 --warmup 5 \
    -b 128 -j 16 \
    --image_size 400 224 \
    --mixup --multi_scale --aug --pretrained \
    --gpus 1 --nodes 1
