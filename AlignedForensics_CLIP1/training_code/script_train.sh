# 数据集路径设置
#DATASET=path to dataset
DATASET="ENTER DATASET PATH"
ours_sync="PATH TO PRETRAINED DETECTOR"
EXP_NAME="SAVE DIR"
#LDM_DS_NAME=directory of checkpoints
# LDM_DS_NAME=检查点目录

# Train on 1000 real and 1000 fake in the sync mode (use half the usual batch size as the code picks a fake image for every real image)
#CUDA_VISIBLE_DEVICES=2 python train.py --name=$LDM_DS_NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET --batch_size 32 --earlystop_epoch 10 --use_inversions --seed 17 --batched_syncing --data_cap=1000
# 同步模式训练（使用1000张真实和1000张生成图像，批量减半以保证每张真实图像对应一张生成图像）

# Train on whole dataset without sync mode
#CUDA_VISIBLE_DEVICES=2 python train.py --name=$LDM_DS_NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET --batch_size 128 --earlystop_epoch 10 --use_inversions --seed 17 
# 非同步全数据集训练

# ICML experiments (Stay-Positive with batch level alignment)
CUDA_VISIBLE_DEVICES=0 python train.py --name=$EXP_NAME --arch res50nodown --cropSize 96  --norm_type resnet --resize_size 256 --resize_ratio 0.75 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_prob 0.2 --jitter_prob  0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --dataroot $DATASET --batch_size 512 --earlystop_epoch 5 --seed 14 --stay_positive "clamp" --ckpt=$ours_sync --fix_backbone --final_dropout 0.0  --use_inversions --batched_syncing 
# ICML实验（使用Stay-Positive方法和批次对齐）
