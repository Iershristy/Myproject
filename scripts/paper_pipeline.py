#!/usr/bin/env python3
"""
Paper pipeline for Figshare PD dataset:
1) Convert raw trials -> skeleton/silhouette multi-view
2) Train multimodal model
3) Evaluate metrics (PD and severity)
4) Save Grad-CAM and part-attention summaries
5) Extract kinematic features and correlate with PD logits
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import subprocess


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    raw_dir = Path('data/raw/pd3d')
    meta_csv = raw_dir / 'meta.csv'
    data_dir = Path('data')

    # 1) Convert
    run(['python', '-m', 'pd_gait.datasets.figshare_convert', '--raw_dir', str(raw_dir), '--metadata_csv', str(meta_csv), '--data_dir', str(data_dir), '--views', '8'])

    # 2) Train
    run(['python', 'train.py', '--config', 'configs/default.yaml'])

    # 3) Evaluate
    run(['python', '-m', 'pd_gait.engine.evaluate', '--config', 'configs/default.yaml', '--ckpt', 'experiments/default/best.pt', '--out', 'experiments/default/metrics.json'])
    print("Metrics:")
    print(Path('experiments/default/metrics.json').read_text())

    print('Artifacts saved under experiments/default/. Grad-CAM in gradcam/val_cam.npy')


if __name__ == '__main__':
    main()

