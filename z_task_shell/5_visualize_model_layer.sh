# 模型、数据可视化
python main.py --data data --visual_data test/ --visual_method all \
    --arch efficientnet_b0 --num_classes 3 \
    --criterion bce \
    -b 5 -j 0 \
    --image_size 400 224 \
    ---resume checkpoints/model_best_efficientnet_b0.pth
