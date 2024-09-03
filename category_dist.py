import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from trainer.RetrievalTrainer import trainer
from datasets.Reader import *


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
        self.catid = "03001627"
        self.voxel_size = 0.03
        self.dim = [1024, 512,  256]
        self.embedding = "conv1_max_embedding"
        self.resume = "./ckpts/cat_ret_conv1max_01_FCGFus"
        self.model = "ResUNetBN2C"
        self.model_n_out = 16
        self.normalize_feature = True
        self.conv1_kernel_size = 3
        self.bn_momentum = 0.05

config = Config()

reader = CategoryLibReader(config.root, config.catid, ["train", "test", "val"], 2000, normal=True)
readerloader = torch.utils.data.DataLoader(reader, batch_size=1, shuffle=False, num_workers=8)
pcs_ref = []
for data in tqdm(readerloader):
    pcs_ref.append(data[0, :, :].cuda())
    

table = compute_dist(pcs_ref)

np.save("/scannet/tables/{}_full".format(config.catid), table)
