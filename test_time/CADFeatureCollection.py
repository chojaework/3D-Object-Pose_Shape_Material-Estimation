import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from trainer.RetrievalTrainer import trainer
from datasets.CategoryTestTimeDataset import *
from datasets.Scan2cadTestTimeDataset import *

from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import *
from model import load_model, fc
from config import get_config

import MinkowskiEngine as ME
    
class CADFeatureCollection():
    """
    Collect global features of CAD models in selected database
    """
    def __init__(self, root, catid, split, distance, batch_size, voxel_size, model, embedding, extra=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = root
        self.voxel_size = voxel_size
        self.split = split
        self.catid = catid
        self.distance = distance
        self.batch_size = batch_size

        if split == "train":
            self.dataset = CategoryTestTimeDataset(self.root, "train", self.catid, self.voxel_size)
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, 
                                                            num_workers=4, collate_fn=self.dataset.collate_pair_fn)
        elif split == "val":
            self.dataset = CategoryTestTimeDataset(self.root, "val", self.catid, self.voxel_size)
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, 
                                                            num_workers=4, collate_fn=self.dataset.collate_pair_fn)
        elif split == "test":
            self.dataset = CategoryTestTimeDataset(self.root, "test", self.catid, self.voxel_size)
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, 
                                                            num_workers=4, collate_fn=self.dataset.collate_pair_fn)
        elif split == "scan2cad":
            # Using CAD models used in Scan2CAD dataset
            scan2cad_dict = extra["scan2cad_dict"]
            self.dataset = Scan2cadTestTimeDataset(self.root, self.catid, scan2cad_dict, self.voxel_size)
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, 
                                                            num_workers=4, collate_fn=self.dataset.collate_pair_fn)
        else:
            raise ValueError("No such split")
            
        self.model = model
        self.embedding = embedding
        
        self.model.eval()
        self.embedding.eval()

    def collect(self):
        return self.eval(self.dataset, self.loader, self.split)

    def eval(self, dataset, dataloader, split):
        """
        Extract global features
        """
        self.model.eval()
        self.embedding.eval()

        base_outputs = []
        base_origins = []

        glob_feats = []

        num_data = len(dataset)
        
        with torch.no_grad():

            for idx, data in enumerate(dataloader):
                if idx%10 == 0:
                    print("Eval feature Index: {}/{}".format(idx+1, len(dataloader)))

                base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")

                batch_size = len(data["base_idx"])

                _, base_feat = self.model(base_input)

                base_feat = self.embedding(base_feat)

                if self.distance == "l2":
                    base_feat_norm = F.normalize(base_feat, dim=1) #base_feat/base_feat.norm(dim=1, keepdim=True)
                    glob_feats.append(base_feat_norm.detach().cpu().numpy())

                else:
                    base_feat_norm = F.normalize(base_feat.F, dim=1) #base_feat/base_feat.norm(dim=1, keepdim=True)

                    for i in range(batch_size):
                        feat_mask = base_feat.C[:,0]==i
                        glob_feats.append(base_feat_norm[feat_mask, :].detach().cpu().numpy())  

        if self.distance == "l2":
            glob_feats = np.concatenate(glob_feats, 0)

        return glob_feats

