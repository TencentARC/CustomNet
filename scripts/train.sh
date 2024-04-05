python main/train.py \
    -t \
    --base configs/config_customnet.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --logdir logs \
    --num_nodes 1 \
    --seed 42
