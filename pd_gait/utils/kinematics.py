from typing import Dict, Any
import numpy as np


def stride_length(ankle_xy: np.ndarray) -> float:
    # approximate stride length from peak-to-peak in x of ankle trajectory
    x = ankle_xy[:, 0]
    return float(np.max(x) - np.min(x))


def arm_swing_amplitude(wrist_xy: np.ndarray, shoulder_xy: np.ndarray) -> float:
    # amplitude of wrist relative to shoulder in x
    rel = wrist_xy[:, 0] - shoulder_xy[:, 0]
    return float(np.percentile(rel, 95) - np.percentile(rel, 5))


def foot_min_clearance(ankle_xy: np.ndarray) -> float:
    # proxy: min y of ankle (after centering) -> lower means more drag
    y = ankle_xy[:, 1]
    return float(np.min(y))


def extract_kinematics(skel: np.ndarray, joint_map: Dict[str, int]) -> Dict[str, float]:
    # skel: [T,J,3] with (x,y,conf) in projected 2D (centered already in model)
    out: Dict[str, float] = {}
    def get(j):
        return skel[:, joint_map[j], :2]
    try:
        out['stride_length_left'] = stride_length(get('l_ankle'))
        out['stride_length_right'] = stride_length(get('r_ankle'))
        out['arm_swing_left'] = arm_swing_amplitude(get('l_wrist'), get('l_shoulder'))
        out['arm_swing_right'] = arm_swing_amplitude(get('r_wrist'), get('r_shoulder'))
        out['foot_min_clearance_left'] = foot_min_clearance(get('l_ankle'))
        out['foot_min_clearance_right'] = foot_min_clearance(get('r_ankle'))
    except Exception:
        pass
    return out

