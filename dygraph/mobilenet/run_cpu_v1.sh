PYTHONPATH=/data/sfraczek/Paddle/build/python DNNL_VERBOSE=1 FLAGS_use_mkldnn=true python train.py --use_gpu=False  --batch_size=64        --total_images=1281167    --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output/ --lr_strategy=piecewise_decay --lr=0.1   --data_dir=/data/ILSVRC2012  --l2_decay=3e-5  --model=MobileNetV1 
