# export FLAGS_use_mkldnn=1
# export OMP_NUM_THREADS=14
# export KMP_AFFINITY=granularity=fine,compact,1,0
# export KMP_BLOCKTIME=1
# export CPU_NUM=1
# time numactl --membind=0 --physcpubind=0-13 \
python eval.py \
       --model=ResNet50 \
       --batch_size=16 \
       --iterations=3125 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --with_mem_opt=False \
       --use_gpu=True \
       --pretrained_model=/mnt/drive/recovered/ResNet50_pretrained
