import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from trainer.RetrievalTrainer import trainer
from datasets.ScannetDataset import ScannetDataset
from datasets.CategoryTestTimeDataset import *
from datasets.Reader import *

from test_time.FeatureExtractor import FeatureExtractor
from test_time.RetrievalModule import RetrievalModule

from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import *
from utils.visualize import *
from utils.preprocess import random_rotation, load_norm_pc, apply_transform
from utils.symmetry import *
from utils.eval_pose import *
from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.Info.CADLib import *
from utils.read_json import build_pcd

from model import load_model, fc

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib
import time

import open3d as o3d
import MinkowskiEngine as ME

torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument("--catid", type=str, default="")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--split", type=str, default="")

args = parser.parse_args()
class Config():
    def __init__(self):

        self.root = "/scannet/ShapeNetCore.v2.PC15k"
        self.scan2cad_root = "/scannet/crop_scan2cad_filter/data"
        self.cad_root = "/scannet/ShapeNetCore.v2.PC15k"
        self.catid = ""
        #self.catid = "04379243"
        #self.catid = "03001627"
        self.resume = ""
        #self.resume = "./ckpts/cat_table_pose_id_01_FCGF16"
        #self.resume = "./ckpts/cat_pose_id_01_FCGF16"
        self.split = ""
        self.voxel_size = 0.03
        self.dim = [1024, 512,  256]
        self.embedding = "identity"
        self.model = "ResUNetBN2C"
        self.model_n_out = 16
        self.normalize_feature = True
        self.conv1_kernel_size = 3
        self.bn_momentum = 0.05
        self.nn_max_n = 500
        self.lib_path = "./CadLib"
        self.scan2cad_dict = "/scannet/scan2cad_download_link/unique_cads.csv"
        self.annotation_dir = "/scannet/scan2cad_download_link"
        self.device = torch.device("cuda")


config = Config()

config.split = args.split
config.resume = args.resume
config.catid = args.catid

logger = logger("./logs", "shapenet-{}-test_log.txt".format(config.catid))
logger.log(config.catid)
logger.log(config.resume)
logger.log(config.split)

dataset = CategoryDataset(config.cad_root, config.split, config.catid, "/scannet/tables", 
                                                     0.1, 0.5, config.voxel_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                        shuffle=False, num_workers=2, collate_fn=dataset.collate_pair_fn)

num_feats = 1
Model = load_model("ResUNetBN2C")
model = Model(
    num_feats,
    16,
    bn_momentum=0.05,
    normalize_feature=True,
    conv1_kernel_size=3,
    D=3)
model = model.to(torch.device("cuda"))


# Embedding network for retrieval
embedding = fc.identity().to("cuda")
distance = "chamfer"


checkpoint = torch.load(config.resume)

model.load_state_dict(checkpoint["state_dict"])

embedding.load_state_dict(checkpoint["embedding_state_dict"])

logger.log(checkpoint["epoch"])

model.eval()
embedding.eval()

np.random.seed(31)
torch.random.manual_seed(31)


########################################################################################


logger.log("start eval")
np.random.seed(31)
torch.random.manual_seed(31)
model.eval()
embedding.eval()

labels = []

base_outputs = [];pos_outputs = []
base_origins = [];pos_origins = []

glob_feats = []

base_Ts = [];pos_Ts = [];
base_syms = [];pos_syms = []
        
        
num_data = len(dataset)

with torch.no_grad():

    for idx, data in enumerate(loader):
        logger.log("Eval feature Index: {}/{}".format(idx+1, len(loader)))

        base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")
        pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to("cuda")

        base_Ts.append(data["base_T"])
        pos_Ts.append(data["pos_T"])

        base_syms.append(data["base_sym"])
        pos_syms.append(data["pos_sym"])

        labels.append(data["labels"])
        batch_size = len(data["labels"])

        base_output, base_feat = model(base_input)

        pos_output, pos_feat = model(pos_input)

        base_feat = embedding(base_feat)

        for i in range(batch_size):
            base_mask = base_output.C[:,0]==i
            pos_mask = pos_output.C[:,0]==i

            base_outputs.append(base_output.F[base_mask, :])
            pos_outputs.append(pos_output.F[pos_mask, :])
            base_origins.append(data["base_origin"][base_mask, :])
            pos_origins.append(data["pos_origin"][pos_mask, :])


        if distance == "l2":
            base_feat_norm = torch.nn.functional.normalize(base_feat, dim=1) 
            glob_feats.append(base_feat_norm)

        else:
            base_feat_norm = torch.nn.functional.normalize(base_feat.F, dim=1) 

            for i in range(batch_size):
                feat_mask = base_feat.C[:,0]==i
                glob_feats.append(base_feat_norm[feat_mask, :])  


    base_Ts = torch.cat(base_Ts, 0)
    pos_Ts = torch.cat(pos_Ts, 0)
    base_syms = torch.cat(base_syms, 0)
    pos_syms = torch.cat(pos_syms, 0)
    


k_nn=5
max_corr = 0.20

with torch.no_grad():
    t_losses_ransac = []
    r_losses_ransac = []

    t_losses_sym = []
    r_losses_sym = []
    
    chamf_ransac = []
    chamf_sym = []
    
    for idx in range(len(dataset)):
        logger.log("----------------------")
        if (idx+1) % 10 == 0:
            
            logger.log("Eval align Index: {}/{}  ".format(idx+1, len(dataset))) 
                            
            logger.log("RANSAC: T error: {:.4f} R error: {:.4f}".format(sum(t_losses_ransac)/len(t_losses_ransac), 
                                                           sum(r_losses_ransac)/len(r_losses_ransac)))
            
            logger.log("SYM: T error: {:.4f} R error: {:.4f}".format(sum(t_losses_sym)/len(t_losses_sym), 
                                                           sum(r_losses_sym)/len(r_losses_sym)))
            logger.log("----------------------")

        # matching based pose estimation
        xyz0, xyz1, T0, T1 = base_origins[idx], pos_origins[idx], base_Ts[idx,:,:], pos_Ts[idx,:,:]
        baseF, posF = base_outputs[idx], pos_outputs[idx]

        base_sym = base_syms[idx].item()
        pos_sym = pos_syms[idx].item()
        
        T_est_best, chamf_dist_best, T_est_ransac, chamf_dist_ransac = sym_pose(baseF, xyz0, posF, 
                                                                                xyz1, pos_sym, k_nn, max_corr)
        
        t_loss_sym, r_loss_sym = eval_pose(T_est_best, T0, T1, axis_symmetry=max(pos_sym, base_sym))

        t_loss_ransac, r_loss_ransac = eval_pose(T_est_ransac, T0, T1, axis_symmetry=max(pos_sym, base_sym))
        
        t_losses_ransac.append(t_loss_ransac)
        r_losses_ransac.append(r_loss_ransac)
        chamf_ransac.append(chamf_dist_ransac)

        logger.log("ransac: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(t_loss_ransac, 
                                                                 r_loss_ransac, 
                                                                 chamf_dist_ransac ))
        
        logger.log("ransac avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_ransac)/len(t_losses_ransac), 
                                                                     sum(r_losses_ransac)/len(r_losses_ransac),
                                                                     sum(chamf_ransac)/len(chamf_ransac)))
        
            
        t_losses_sym.append(t_loss_sym)
        r_losses_sym.append(r_loss_sym)
        chamf_sym.append(chamf_dist_best)

        logger.log("sym: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(t_loss_sym, 
                                                              r_loss_sym, 
                                                              chamf_dist_best))
        
        logger.log("sym avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_sym)/len(t_losses_sym), 
                                                                  sum(r_losses_sym)/len(r_losses_sym),
                                                                  sum(chamf_sym)/len(chamf_sym)))
        logger.log("----------------------")

        
             
                
                
np.save("/zty-vol/results/stats/shapenet-{}-T-ransac.npy".format(config.catid), np.array(t_losses_ransac))
np.save("/zty-vol/results/stats/shapenet-{}-R-ransac.npy".format(config.catid), np.array(r_losses_ransac))

np.save("/zty-vol/results/stats/shapenet-{}-T-sym.npy".format(config.catid), np.array(t_losses_sym))
np.save("/zty-vol/results/stats/shapenet-{}-R-sym.npy".format(config.catid), np.array(r_losses_sym))


np.save("/zty-vol/results/stats/shapenet-{}-chamf-ransac.npy".format(config.catid), np.array(chamf_ransac))
np.save("/zty-vol/results/stats/shapenet-{}-chamf-sym.npy".format(config.catid), np.array(chamf_sym))


logger.log("ransac avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_ransac)/len(t_losses_ransac), 
                                                             sum(r_losses_ransac)/len(r_losses_ransac),
                                                             sum(chamf_ransac)/len(chamf_ransac)))

logger.log("sym avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_sym)/len(t_losses_sym), 
                                                                  sum(r_losses_sym)/len(r_losses_sym),
                                                                  sum(chamf_sym)/len(chamf_sym)))