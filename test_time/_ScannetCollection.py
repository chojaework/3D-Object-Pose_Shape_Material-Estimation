import os
import torch
import csv
import random
from tqdm import tqdm
import numpy as np
import open3d as o3d
import transforms3d
import MinkowskiEngine as ME

from utils.preprocess import *
from utils.visualize import * 
from utils.read_json import *
from datasets.Reader import *
from dataset.ScannetDataset import *

class ScannetTestTimeDataset(ScannetDataset):
    
    def quant(self, rot_coords, coords):
        unique_idx = ME.utils.sparse_quantize(np.floor(rot_coords / self.voxel_size), return_index=True) 
        rot_coords = rot_coords[unique_idx, :]
        coords = coords[unique_idx, :]
        rot_coords_grid = np.floor(rot_coords / self.voxel_size)

        return rot_coords, rot_coords_grid, coords    def __len__(self):
        return len(self.pcs)
    
    def __getitem__(self, idx):
        
        
        
        scan_points = apply_trans(self.pcs[idx], self.ScanPoses[idx]["translation"], self.ScanPoses[idx]["rotation"],
                                  self.ScanPoses[idx]["scale"], mode="normal") 

        cad_path = self.id2path[self.BestMatches[idx]]
        print(cad_path)
        cad_pc = load_raw_pc(os.path.join(self.cad_root, cad_path), 10000)
        
        cad_pc = apply_trans(cad_pc, self.CadPoses[idx]["translation"], self.CadPoses[idx]["rotation"], 
                    self.CadPoses[idx]["scale"], mode="normal")
        
        scan_points -= cad_pc.mean(0)
        cad_pc -= cad_pc.mean(0)
        
        base_coords = scan_points
        
        rot_base_coords, rot_base_coords_grid, base_coords = self.quant(base_coords, base_coords)

        base_feat = np.ones([len(rot_base_coords), 1])


        # tag: data output
        base = {"coord": rot_base_coords_grid, "origin":rot_base_coords, "feat": base_feat, "idx":idx}

        
        return base
        #return scan_points, cad_pc


        
    def collate_pair_fn(self, list_data):
        #print(type(list_data))
        #print(len(list_data))
        
        base_dict = list_data

        base_coords = []
        base_feat = []
        base_T = []
        base_origin = []
        base_idx = []

        for idx in range(len(base_dict)):

            base_coords.append(torch.from_numpy(base_dict[idx]["coord"]))
            base_origin.append(torch.from_numpy(base_dict[idx]["origin"]))
            base_feat.append(torch.from_numpy(base_dict[idx]["feat"]))
            base_idx.append(base_dict[idx]["idx"])        
   
        
        batch_base_coords, batch_base_feat = ME.utils.sparse_collate(base_coords, base_feat)

        data = {}
        
        data["base_coords"] = batch_base_coords.int()
        data["base_feat"] = batch_base_feat.float()
        
        data["base_origin"] = torch.cat(base_origin, 0).float()

        data["base_idx"] = torch.tensor(base_idx)
        
        return data
