python3 train.py --dataroot /dev/shm --dataset multivkitti --name sasn --nepoch 100 --gpu_ids 0 --batch_size 16 --make_name demo --lr_policy lambda --lr 0.05 --useWidth 320 --useHeight 320 --use_sne --sne d2ntv2 --save_period 100 --split_scheme ./datasets/all_120
