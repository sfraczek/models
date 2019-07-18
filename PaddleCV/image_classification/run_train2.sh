# export FLAGS_use_mkldnn=1
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export CPU_NUM=1
       # --total_images=1281167 \
       # --num_epochs=120 \
time numactl --membind=0 --physcpubind=0-13 \
python train2.py \
       --model=ResNet50 \
       --batch_size=16 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=False \
       --use_gpu=True \
       --total_images=1281167 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=1 \
       --lr=0.1 \
       --l2_decay=1e-4
