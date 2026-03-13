#!/bin/bash

# USB 카메라 실시간 이상 탐지 실행 스크립트

# 1. 환경 설정
CAM_ID=0             # /dev/video0
INTERVAL=30          # 촬영 간격 (초)
THRESHOLD=0.5        # 이상 탐지 임계값 (0.0 ~ 1.0)
SAVE_DIR="./results/realtime"
CHECKPOINT="./checkpoints/9_12_4_multiscale/epoch_15.pth"

# 2. 실행
echo "--- Starting AnomalyCLIP Real-time USB Camera Inference ---"
echo "Camera Index: ${CAM_ID}"
echo "Interval: ${INTERVAL}s"
echo "Threshold: ${THRESHOLD}"
echo "--------------------------------------------------------"

python usb_camera_inference.py \
    --cam_id ${CAM_ID} \
    --interval ${INTERVAL} \
    --threshold ${THRESHOLD} \
    --save_path ${SAVE_DIR} \
    --checkpoint_path ${CHECKPOINT} \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4

