# 课程学习时，制作不同样本的损失函数权重文件
python main.py --data data --make_curriculum train \
    --curriculum_thresholds 0.7 0.5 0.3 0.0 \
    --curriculum_weights 1 0.7 0.4 0.1 \
    --criterion bce \
    --image_size 400 224 -b 256 -j 16 \
    ---resume checkpoints/model_best_efficientnet_b0.pth \
    -g 1 -n 1
