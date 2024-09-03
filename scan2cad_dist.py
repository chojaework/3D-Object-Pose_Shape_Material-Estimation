import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from trainer.RetrievalTrainer import trainer
from datasets.CategoryTestTimeDataset import *
from datasets.Reader import *

from utils.Info.Scan2cadInfo import Scan2cadInfo
from utils.Info.CADLib import CustomizeCADLib
from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import *
from utils.visualize import *
from utils.pc_dist import *

from model import load_model, fc

import MinkowskiEngine as ME

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Config():
    def __init__(self):

        self.root = "/scannet/ShapeNetCore.v2.PC15k"
        self.scan2cad_root = "/scannet/crop_scan2cad_filter/data"
        self.cad_root = "/scannet/ShapeNetCore.v2.PC15k"
        self.catid = "04379243"
        self.voxel_size = 0.03
        self.dim = [1024, 512,  256]
        self.embedding = "identity"
        self.resume = "./ckpts/cat_table_pose_id_01_FCGF16"
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

scan2cad_info = Scan2cadInfo(config.cad_root, config.scan2cad_root, config.catid, config.annotation_dir)

CadReader = ReaderWithPath(scan2cad_info.UsedObjPath, 2000, normal=True)

readerloader = torch.utils.data.DataLoader(CadReader, batch_size=1, shuffle=False, num_workers=8)
pcs_ref = []
for data in tqdm(readerloader):
    pcs_ref.append(data[0, :, :].cuda())
    

table = compute_dist(pcs_ref)

np.save("/scannet/tables/{}_scan2cad".format(config.catid), table)