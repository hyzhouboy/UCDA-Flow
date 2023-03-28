#!/bin/bash
mkdir -p checkpoints

# stage 1
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage kitti --restore_flow_ckpt model/init/raft.pth --restore_disp_ckpt model/init/aanet.pth \
#     --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision --load_pseudo_gt

# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage kitti --restore_flow_ckpt model//init/raft.pth --restore_disp_ckpt model/init/aanet.pth \
#     --gpus 0 1 --num_steps 100000 --batch_size 4 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision

# stage 2 
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage kitti --restore_flow_ckpt model/raft.pth --restore_disp_ckpt model/aanet.pth \
#     --restore_pose_encoder_ckpt model/pose/pose_encoder.pth --restore_pose_decoder_ckpt model/pose/pose.pth \
#     --gpus 0 --num_steps 100000 --batch_size 2 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 3 
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage kitti --restore_flow_ckpt model/raft.pth --restore_disp_ckpt model/aanet.pth \
#     --gpus 0 --num_steps 100000 --batch_size 1 --lr 0.0001 --image_size 375 1242 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 4 
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage kitti_foggy --restore_flow_ckpt model/raft.pth --restore_flow_synimg_ckpt model/ucda_flow_synfog.pth \
#     --gpus 0 --num_steps 100000 --batch_size 4 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision


# stage 5
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage real_foggy --restore_flow_synimg_ckpt model/ucda_flow_synfog.pth --restore_flow_realimg_ckpt model/ucda_flow_realfog.pth \
#     --gpus 0 --num_steps 100000 --batch_size 3 --lr 0.0001 --image_size 288 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision

# stage 6
# python -u main.py --data_dir /home/sdb/zhouhanyu/data \
#     --stage real_foggy --restore_flow_synimg_ckpt model/ucda_flow_synfog.pth --restore_flow_realimg_ckpt model/ucda_flow_realfog.pth \
#     --gpus 0 --num_steps 100000 --batch_size 3 --lr 0.0001 --image_size 288 640 --wdecay 0.00001 --gamma=0.85 --mixed_precision

# stage 7
python -u main.py --data_dir /home/sdb/zhouhanyu/data \
    --stage real_foggy --restore_flow_synimg_ckpt model/ucda_flow_synfog.pth --restore_flow_realimg_ckpt model/ucda_flow_realfog.pth \
    --gpus 0 --num_steps 100000 --batch_size 2 --lr 0.0001 --image_size 288 512 --wdecay 0.00001 --gamma=0.85 --mixed_precision --use_context_attention

