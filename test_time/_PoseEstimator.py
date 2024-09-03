import torch
import os

from utils.symmetry import symmetric_cut
from utils.eval_pose import estimate_pose, estimate_pose_combine

class PoseEstimator:
    def __init__(self):
        pass

    def estimate(self, xyz0, xyz1, feat0, feat1, nn_max_n, symmetry=True):
        if symmetry:

            base_v, base_d, base_half0_mask, base_half1_mask = symmetric_cut(feat0, xyz0)
            pos_v, pos_d, pos_half0_mask, pos_half1_mask = symmetric_cut(feat1, xyz1)

            avgdist00, T_est00 = estimate_pose_combine(xyz0[base_half0_mask], xyz1[pos_half0_mask],
                            feat0[base_half0_mask], feat1[pos_half0_mask],
                            xyz0[base_half1_mask], xyz1[pos_half1_mask],
                            feat0[base_half1_mask], feat1[pos_half1_mask], nn_max_n)

            avgdist11, T_est01 = estimate_pose_combine(xyz0[base_half0_mask], xyz1[pos_half1_mask],
                            feat0[base_half0_mask], feat1[pos_half1_mask],
                            xyz0[base_half1_mask], xyz1[pos_half0_mask],
                            feat0[base_half1_mask], feat1[pos_half0_mask], nn_max_n)

            if avgdist00 < avgdist11:
                T_est = T_est00
            else:
                T_est = T_est01

        else:
            avg_dist, T_est = estimate_pose(xyz0, xyz1, feat0, feat1, nn_max_n)
  
        return T_est