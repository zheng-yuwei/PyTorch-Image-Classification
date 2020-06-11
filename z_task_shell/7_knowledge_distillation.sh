# 模型蒸馏时，准备教师模型的概率文件
python main.py --data data --knowledge train \
    --arch efficientnet_b0 --num_classes 3 \
    --criterion bce \
    --image_size 400 224 -b 256 -j 16 \
    ---resume checkpoints/model_best_efficientnet_b0.pth \
    -g 1 -n 1
