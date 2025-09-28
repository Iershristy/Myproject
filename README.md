Multimodal Deep Learning for Parkinson's Disease Detection from Gait (Occlusion + Multi-View)

Overview
This repository provides a PyTorch implementation for multimodal gait-based Parkinson's Disease (PD) detection and severity estimation under occlusions and multi-view settings. It fuses skeleton keypoints, silhouettes, and optionally IMU signals. The design includes:
- Part-based skeleton encoder inspired by GaitPart with occlusion-robust masking
- Set-based silhouette encoder with Horizontal Pyramid Pooling (HPP) adapted from GaitSet/GaitGL
- Lightweight IMU temporal encoder
- Attention-based modality fusion handling missing modalities
- Dual heads for PD classification and severity levels (mild/moderate/severe)
- Grad-CAM and saliency visualization for interpretability (stride length, arm swing, foot drag)

Quick Start
1) Install
   pip install -r requirements.txt

2) Data layout (example)
   data/
     labels_train.csv  # sample_id,pd_label,severity_label,view_id
     labels_val.csv
     skeleton/
       train/<sample_id>.npy   # [T,J,C]
       val/<sample_id>.npy
     silhouette/
       train/<sample_id>/frame_0001.png ...
       val/<sample_id>/frame_0001.png ...
     imu/
       train/<sample_id>.npy   # [T,S]
       val/<sample_id>.npy

3) Configure
   Edit configs/default.yaml

4) Train
   python train.py --config configs/default.yaml

Notes
- Missing modalities and occlusions are handled via masks and attention gating.
- Severity estimation is optional when labels are present.
- Multi-view robustness is encouraged via set pooling and view-agnostic encoders.

Datasets
- PD 3D Kinematics/Kinetics (best fit):
  - Place raw motion-capture trials under `data/raw/pd3d/` (requires access approval). Then run:
    python -m pd_gait.datasets.pd3d_convert --raw_dir data/raw/pd3d --data_dir data
  - If a public mirror becomes available, try the helper (currently placeholder):
    python -m pd_gait.datasets.pd3d_downloader --raw_dir data/raw/pd3d --try_public_url
- Daphnet FOG IMU (optional IMU robustness):
  python -m pd_gait.datasets.daphnet_downloader --out_dir data/raw/daphnet
  (You can later write a small converter to align IMU files to labels and copy to `data/imu/`.)

