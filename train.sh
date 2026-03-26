
# CUDA_VISIBLE_DEVICES=6,7 python3 multi_train.py
# sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && sudo cp /home/weidong.shi1/.conda/pkgs/libstdcxx-ng-13.2.0-h7e041cc_3/lib/libstdc++.so.6.0.32 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=12395 multi_train.py