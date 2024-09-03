import os
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import transforms3d
import MinkowskiEngine as ME

from trainer.RetrievalTrainer import trainer

from model import load_model, fc

class FeatureExtractor(trainer):
    """
    Global and local Feature extractor
    """
    def __init__(self, config):
        
        print("Feature Extraction")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.root = self.config.root 
        self.voxel_size = self.config.voxel_size
        
        print("Embedding: {}".format(self.config.embedding))
        print("FC dim: {}".format(self.config.dim))
        print("voxel size: {}".format(self.voxel_size))
        
        # Model initialize
        # FCGF
        num_feats = 1
        Model = load_model(self.config.model)
        model = Model(
            num_feats,
            self.config.model_n_out,
            bn_momentum=self.config.bn_momentum,
            normalize_feature=self.config.normalize_feature,
            conv1_kernel_size=self.config.conv1_kernel_size,
            D=3)
        self.model = model.to(self.device)
        
        # Embedding network for retrieval
        if self.config.embedding == "conv1":
            assert len(self.config.dim) == 1
            self.embedding = fc.conv1_chamfer(self.config.dim[0]).to(self.device)
            self.distance = "chamfer"
        elif self.config.embedding == "identity":
            self.embedding = fc.identity().to(self.device)
            self.distance = "chamfer"
        elif self.config.embedding == "netvlad":
            self.embedding  = fc.NetVLAD().to(self.device)
            self.distance = "l2"
        elif self.config.embedding == "max_embedding":
            assert len(self.config.dim) == 2
            linear1dim, linear2dim = self.config.dim
            self.embedding  = fc.max_embedding(256, linear1dim, linear2dim).to(self.device)
            self.distance = "l2"
        elif self.config.embedding == "conv1_max_embedding":
            assert len(self.config.dim) == 3
            conv_dim, linear1dim, linear2dim = self.config.dim
            self.embedding  = fc.conv1_max_embedding(conv_dim, linear1dim, linear2dim).to(self.device)
            self.distance = "l2"
        else:
            raise ValueError("Embedding model {} not defined".format(self.config.embedding))
        if self.distance == "chamfer":
            raise ValueError("Not implemented")
        
        # Must resume from previous checkpoint
        if self.config.resume:
            print("loading checkpoint from {}".format(self.config.resume))
            checkpoint = torch.load(self.config.resume)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.embedding.load_state_dict(checkpoint["embedding_state_dict"])
            start_epoch = checkpoint["epoch"]
            print("Model at epoch: {}".format(start_epoch))
        else:
            raise ValueError("Must have a pretrained model (use resume)")
            
        self.model.eval()
        self.embedding.eval()

    def quant(self, rot_coords):
        """
        Point cloud quantize
        """
        unique_idx = ME.utils.sparse_quantize(torch.floor(rot_coords / self.voxel_size), return_index=True) 
        rot_coords = rot_coords[unique_idx, :]
        rot_coords_grid = torch.floor(rot_coords / self.voxel_size)

        return rot_coords, rot_coords_grid

    def process(self, coords):
        """
        Global and local feature extraction
        Input:
            coordinates
        Output:
            local feature, global featyre, quantized coordinates
        """
        if isinstance(coords, np.ndarray):
            coords = torch.Tensor(coords)
        
        coords, coords_grid = self.quant(coords)
        coords_grid = ME.utils.batched_coordinates([coords_grid.int()])
        feats = torch.ones([len(coords), 1]).float()

        input = ME.SparseTensor(feats=feats, coords=coords_grid).to(self.device)

        local_feature, global_feature = self.model(input)
        global_feature = self.embedding(global_feature)
        global_feature = F.normalize(global_feature, dim=1)
        global_feature = global_feature.squeeze()

        return local_feature.F, global_feature, coords



