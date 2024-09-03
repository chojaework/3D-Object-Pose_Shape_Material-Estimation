import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
##### batch size #####
trainer_arg.add_argument('--batch_size', type=int, default=32)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetBN2C')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=7)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument(
    '--icp_cache_path', type=str, default="/home/chrischoy/datasets/FCGF/kitti/icp/")




# Dataset specific configurations
data_arg = add_argument_group('Data')

data_arg.add_argument('--resume', type=str, default=None, help="Checkpoint to resume")
data_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=500,
    help='The maximum number of features to find nearest neighbors in batch')
data_arg.add_argument('--voxel_size', type=float, default=0.03, help="Voxel size for quantize")

data_arg.add_argument('--log_name', type=str, default='debug_log', help="Pog path")
data_arg.add_argument('--ckpt_name', type=str, default='debug_ckpt', help="Pytorch checkpoint name")
data_arg.add_argument('--pretrain', type=str, default='', help="Used pretrained model")
data_arg.add_argument('--mode', type=str, default='train', help="train/val/test mode will influence the data loading and model resuming")

data_arg.add_argument('--root', type=str, default='', help="Shapenet dir")
data_arg.add_argument('--scan2cad_root', type=str, default='', help="Cropped scan2cad objects dir")
data_arg.add_argument('--preload', type=str2bool, default=True, help="Load all point clouds to memory before training")

data_arg.add_argument('--dataset', type=str, default='', help="Dataset selected")
data_arg.add_argument('--catid', type=str, default='', help="Category Id selected")

data_arg.add_argument('--dim', nargs='+', type=int, help="Embedding fully connected layers dimension")

data_arg.add_argument('--eval_epoch', type=int, default=5, help="Eval every N epoch")
data_arg.add_argument('--log_batch', type=int, default=5, help="Log every N epoch")

data_arg.add_argument('--pos_ratio', type=float, default=0.1, help="Ratio of cad models considered as positive")
data_arg.add_argument('--neg_ratio', type=float, default=0.5, help="Ratio of cad models considered as negative")

data_arg.add_argument('--train_ret', default=False, action='store_true', help="Train embedding for retrieval")
data_arg.add_argument('--train_pose', default=False, action='store_true', help="Train FCGF for pose")

data_arg.add_argument('--embedding', type=str, default="", help="Choose a embedding network")

data_arg.add_argument('--use_symmetry', default=False, action='store_true', help="Use symmetry in pose estimation")

data_arg.add_argument('--annotation_dir', default='', type=str, help="Scan2cad annotation dir")


def get_config():
  args = parser.parse_args()
  return args
