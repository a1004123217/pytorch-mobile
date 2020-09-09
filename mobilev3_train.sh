python3 imagenet.py \
--arch MobileNetV3 \
--data ../imagenet/ \
--epochs 30 --schedule 11 21  --lr 0.0005 --gpu-id 0,1 \
--model_path './models_init/mbv3_large.pth.tar' \
--checkpoint checkpoints_MobileNetV3_Large/imagenet/MobileNetV3

