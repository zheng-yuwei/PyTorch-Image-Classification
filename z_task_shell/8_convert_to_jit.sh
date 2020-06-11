# 将指定的模型文件转为JIT格式
python main.py --jit --arch efficientnet_b0 --num_classes 3 --image_size 400 224 \
    --resume checkpoints/model_best_efficientnet_b0.pth -g 0
