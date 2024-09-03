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

from model import load_model, fc

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib
import time

import MinkowskiEngine as ME

parser = argparse.ArgumentParser()

parser.add_argument("--catid", type=str, default="")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--retrieve_cad", action="store_true", default=False)

args = parser.parse_args()

class Config():
    def __init__(self):

        self.root = "/scannet/ShapeNetCore.v2.PC15k"
        self.scan2cad_root = "/scannet/crop_scan2cad_filter/data"
        self.cad_root = "/scannet/ShapeNetCore.v2.PC15k"
        #self.catid = "03001627"
        #self.catid = "04379243"
        self.catid = ""
        self.voxel_size = 0.03
        #self.dim = [1024, 512,  256]
        #self.embedding = "identity"
        self.resume = ""
        #self.resume = "./ckpts/scannet_table_ret_max_sim0"
        #self.resume = "./ckpts/scannet_ret_max_sim0"
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
        self.retrieve_cad = False


def eval_lib(model, embedding, cadlib_loader, distance):
    np.random.seed(31)
    torch.random.manual_seed(31)
    model.eval()
    embedding.eval()

    lib_feats = []
    lib_outputs = []
    lib_Ts = []
    lib_origins = []

    with torch.no_grad():
        print("Updating global feature in the CAD Lib")
        for idx, data in enumerate(cadlib_loader):
            print("Lib feature Batch: {}/{}".format(idx+1, len(cadlib_loader)))

            base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")

            lib_Ts.append(data["base_T"])
            batch_size = len(data["base_idx"])

            base_output, base_feat = model(base_input)
            base_feat = embedding(base_feat)

            for i in range(batch_size):
                base_mask = base_output.C[:,0]==i

                lib_outputs.append(base_output.F[base_mask, :])
                lib_origins.append(data["base_origin"][base_mask, :])

            if distance == "l2":
                base_feat_norm = F.normalize(base_feat, dim=1) 
                lib_feats.append(base_feat_norm)

            else:
                base_feat_norm = F.normalize(base_feat.F, dim=1)

                for i in range(batch_size):
                    base_feat_mask = base_feat.C[:,0] == i
                    lib_feats.append(base_feat_norm[base_feat_mask, :])  


    lib_Ts = torch.cat(lib_Ts, 0)

    return lib_feats, lib_outputs, lib_origins, lib_Ts
        
##################################################################################################################

config = Config()

config.resume = args.resume
config.catid = args.catid
config.retrieve_cad = args.retrieve_cad

logger = logger("./logs", "scan2cad-test-{}-{}-log.txt".format(config.catid, config.retrieve_cad))

logger.log("catid: {}".format(config.catid))
logger.log("use retrieve cad: {}".format(config.retrieve_cad))
logger.log("resume: {}".format(config.resume))


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
distance = "l2"#"chamfer"
embedding = fc.conv1_max_embedding(1024, 512, 256).to("cuda")

checkpoint = torch.load(config.resume)
model.load_state_dict(checkpoint["state_dict"])
embedding.load_state_dict(checkpoint["embedding_state_dict"])

logger.log("ckpt epoch: {}".format(checkpoint["epoch"]))

model.eval()
embedding.eval()

# random seed
np.random.seed(31)
torch.random.manual_seed(31)

# dataset
scan2cad_info = Scan2cadInfo(config.cad_root, config.scan2cad_root, config.catid, config.annotation_dir)

table_path = "/scannet/tables/{}_scan2cad.npy".format(config.catid)
cadlib = CustomizeCADLib(config.cad_root, config.catid, scan2cad_info.UsedObjId, table_path, 
                              config.voxel_size, False)

cadlib_loader = torch.utils.data.DataLoader(cadlib, batch_size=32, 
                                        shuffle=False, num_workers=4, collate_fn=cadlib.collate_pair_fn)

dataset = ScannetDataset(config.scan2cad_root, config.cad_root, cadlib, scan2cad_info, "test", config.catid, 
                                            0.0, 0.5, 0.03, preload=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                        shuffle=False, num_workers=2, collate_fn=dataset.collate_pair_fn)

# load sym label
with open("./config/{}_{}_rot_sym_label.txt".format(config.catid, "scan2cad"), "r") as f:

    lines = f.readlines()

    names = [line.strip('\n').split(' ')[0] for line in lines]
    sym_ref = [int(line.strip('\n').split(' ')[1]) for line in lines]

sym_label = sym_ref

# cad models feature extraction
lib_feats, lib_outputs, lib_origins, lib_Ts  = eval_lib(model, embedding, cadlib_loader, distance)
logger.log("lib size: {}".format(len(lib_feats)))

# scans feature extraction
with torch.no_grad():
    np.random.seed(31)
    torch.random.manual_seed(31)
    model.eval()
    embedding.eval()

    t_loss_avg = 0
    r_loss_avg = 0

    pos_idx = []

    base_outputs = []
    base_origins = []

    base_feats = []

    base_Ts = []
    
    pos_syms = []

    num_data = len(dataset)

    # Collect Lib features
    with torch.no_grad():

        for idx, data in enumerate(loader):
            logger.log("Eval feature Batch: {}/{}".format(idx+1, len(loader)))

            base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")

            base_Ts.append(data["base_T"])
            pos_idx.append(data["pos_idx"])
            pos_syms.append(data["pos_sym"])
            batch_size = len(data["pos_idx"])

            base_output, base_feat = model(base_input)

            base_feat = embedding(base_feat)

            for i in range(batch_size):
                base_mask = base_output.C[:,0]==i

                base_outputs.append(base_output.F[base_mask, :])
                base_origins.append(data["base_origin"][base_mask, :])


            if distance == "l2":
                base_feat_norm = F.normalize(base_feat, dim=1) 
                base_feats.append(base_feat_norm)

            else:
                base_feat_norm = F.normalize(base_feat.F, dim=1) 

                for i in range(batch_size):
                    base_feat_mask = base_feat.C[:,0] == i
                    base_feats.append(base_feat_norm[base_feat_mask, :])  

        pos_idx = torch.cat(pos_idx, 0)
        base_Ts = torch.cat(base_Ts, 0)
        pos_syms = torch.cat(pos_syms, 0)

# retrieval part
if distance == "chamfer":
    dists = np.zeros([len(base_feats), len(lib_feats)])
    for i in range(len(base_feats)):
        for j in range(len(lib_feats)):
            dists[i,j] = chamfer_gpu(base_feats[i], lib_feats[j])
    #dists += dists.T
    BestMatchIdx = pos_idx.detach().cpu().numpy()

    stat = scan2cad_retrieval_eval_dist(dists, dataset.table, BestMatchIdx, max(dataset.pos_n, 1))
    logger.log("top1_error: {} percision: {}".format(stat["top1_error"], stat["percision"]))
else:
    descriptor = torch.cat(base_feats, 0).detach().cpu().numpy()
    lib_feats = torch.cat(lib_feats, 0).detach().cpu().numpy()

    BestMatchIdx = pos_idx.detach().cpu().numpy()
    stat = scan2cad_retrieval_eval(descriptor, lib_feats, BestMatchIdx, dataset.table, max(dataset.pos_n, 1))
    logger.log("top1_error: {} percision: {}".format(stat["top1_error"], stat["percision"]))

gt = stat["gt"]
top1_predict = stat["top1_predict"]

# pose estimation
if config.retrieve_cad:
    select = top1_predict
    logger.log("Using top-1 prediction")
else:
    select = gt
    logger.log("Using annotation")

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

        # pose estimation
        xyz0, xyz1, T0, T1 = base_origins[idx], lib_origins[select[idx]], base_Ts[idx,:,:], lib_Ts[select[idx],:,:]
        baseF, posF = base_outputs[idx], lib_outputs[select[idx]]

        pos_sym = sym_label[select[idx]]
                
        T_est_best, chamf_dist_best, T_est_ransac, chamf_dist_ransac = sym_pose(baseF, xyz0, posF, 
                                                                                xyz1, pos_sym, k_nn, max_corr)
        
        t_loss_sym, r_loss_sym = eval_pose(T_est_best, T0, T1, axis_symmetry=pos_sym)

        t_loss_ransac, r_loss_ransac = eval_pose(T_est_ransac, T0, T1, axis_symmetry=pos_sym)
        
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

            
            
np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-T-ransac.npy".format(config.catid, config.retrieve_cad), 
                                                                         np.array(t_losses_ransac))
np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-R-ransac.npy".format(config.catid, config.retrieve_cad),
                                                                        np.array(r_losses_ransac))

np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-T-sym.npy".format(config.catid, config.retrieve_cad),
                                                                        np.array(t_losses_sym))
np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-R-sym.npy".format(config.catid, config.retrieve_cad),
                                                                        np.array(r_losses_sym))

np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-chamf-ransac.npy".format(config.catid, config.retrieve_cad),
                                                                        np.array(chamf_ransac))
np.save("/zty-vol/results/stats/scan2cad-test-{}-{}-chamf-sym.npy".format(config.catid, config.retrieve_cad),
                                                                        np.array(chamf_sym))


logger.log("ransac avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_ransac)/len(t_losses_ransac), 
                                                             sum(r_losses_ransac)/len(r_losses_ransac),
                                                             sum(chamf_ransac)/len(chamf_ransac)))

logger.log("sym avg: t: {:.4f} r: {:.4f} chamf: {:.4f}".format(sum(t_losses_sym)/len(t_losses_sym), 
                                                                  sum(r_losses_sym)/len(r_losses_sym),
                                                                  sum(chamf_sym)/len(chamf_sym)))