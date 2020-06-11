# 用多/单GPU评估模型（没GPU则自动退化为CPU）
python main.py --data data/ -e --arch efficientnet_b0 --num_classes 3 \
    --criterion bce --image_size 400 224 \
    --batch_size 256 -j 16 \
    --resume checkpoints/model_best_efficientnet_b0.pth -g 1 -n 1
