# 训练时，先预估最大学习率
CUDA_VISIBLE_DEVICES=0 python main.py --data data/ --train \
    --arch efficientnet_b0 --num_classes 3 \
    --criterion bce \
    --opt sgd \
    --lr_ratios 0.01 0.1 1. 10. 100. 1.e3 1.e4 1.e5 1.e6 \
    --lr_steps 3 6 9 12 15 18 21 24 27 \
    --epoches 27 --warmup -1 \
    -b 128 -j 16 \
    --image_size 400 224 \
    --aug --pretrained \
    --gpus 1 --nodes 1
