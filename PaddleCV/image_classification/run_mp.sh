THREAD_PER_INSTANCE=12
SOCKET_THREADS=14
for i in {0..7}
do

THR_AFF_LOW_BOUND=$(( $i*$SOCKET_THREADS ))
THR_AFF_HIG_BOUND=$(( $i*$SOCKET_THREADS + $THREAD_PER_INSTANCE - 1 ))

echo "${i} thread affinity bounds: ${THR_AFF_LOW_BOUND}-${THR_AFF_HIG_BOUND}"

python train_aeon.py \
        --model=FakeNet \
        --batch_size=128 \
        --class_dim=1000 \
        --image_shape=3,224,224 \
        --with_mem_opt=True \
        --use_gpu=False \
        --total_images=1281167 \
        --model_save_dir=output/ \
        --lr_strategy=piecewise_decay \
        --num_epochs=1 \
        --lr=0.1 \
        --l2_decay=1e-4 \
        --data_dir=/root/data/ILSVRC2012/ \
        --cache_dir=/root/paddle/.aeon_cache/ \
        --reader_thread_count=$THREAD_PER_INSTANCE \
        --thread_affinity_low_bound=$THR_AFF_LOW_BOUND \
        --thread_affinity_high_bound=$THR_AFF_HIG_BOUND \
        --random_seed=0 &
done
