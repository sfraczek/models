# export FLAGS_use_mkldnn=1
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export CPU_NUM=1
# gdb -ex 'break /mnt/drive/aeon/src/augment_image.cpp:223' --args 
       # --total_images=1281167 \
       # --num_epochs=120 \
time numactl --membind=0 --physcpubind=0-13 \
python train_aeon.py  2>&1 | tee test_of_train_aeon_log.txt
