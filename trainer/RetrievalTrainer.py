"""
log config: log_name

save config: ckpt_name

basic config: trainer, dataset

eval & display: eval_epoch, log_batch

data config: voxel_size, root, catid, pos_ratio, neg_ratio

model config: model, model_n_out, bn_momentum, normalize_feature, conv1_kernel_size, embedding, dim

training config: max_epoch, batch_size, train_ret, train_pose, mode, optimizer, lr, momentum, weight_decay, exp_gamma, resume, pretrain

pose estimation config: nn_max_n

"""

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from datasets.CategoryDataset import CategoryDataset

from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import *
from utils.preprocess import apply_transform
from utils.eval_pose import *
from model import load_model, fc
from config import get_config

import MinkowskiEngine as ME

class trainer:
    def __init__(self, ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = get_config()
        self.logger = logger("./logs", self.config.log_name+".txt")

        self.root = self.config.root #"/zty-vol/data/ShapeNetCore.v2.PC15k"
        self.save_name = self.config.ckpt_name

        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch
        self.eval_iter = self.config.eval_epoch
        self.log_batch = self.config.log_batch
        self.voxel_size = self.config.voxel_size

        if not os.path.exists("./ckpts"):
            os.mkdir("./ckpts")
            
        self.logger.log(self.device)
        self.logger.log("trainer: {}".format(self.config.trainer))
        self.logger.log("mode: {}".format(self.config.mode))
        self.logger.log("category ID: {}".format(self.config.catid))
        self.logger.log("dataset: {}, batch_size: {}".format(self.config.dataset, self.config.batch_size))
        self.logger.log("embedding dim: {}".format(self.config.dim))
        self.logger.log("voxel size: {}".format(self.voxel_size))
        self.logger.log("optimizer: {} learning rate: {}, exp_gamma: {}".format(self.config.optimizer, self.config.lr, self.config.exp_gamma))
        self.logger.log("embedding: {}".format(self.config.embedding))
        
        self.logger.log("pos ratio: {}, neg ratio: {}".format(self.config.pos_ratio, self.config.neg_ratio))
        self.logger.log("train_pose: {}, train_ret: {}".format(self.config.train_pose, self.config.train_ret))
        
        self.logger.log("resume: {}".format(self.config.resume))
        self.logger.log("pretrain: {}".format(self.config.pretrain))
        
    
class RetrievalTrainer(trainer):

    def __init__(self, ):
        trainer.__init__(self)

        # dataset and dataloader
        if self.config.dataset == "CategoryDataset":
            self.pos_ratio = self.config.pos_ratio
            self.neg_ratio = self.config.neg_ratio
            self.catid = self.config.catid
            self.logger.log("CatId: {}".format(self.catid))
            self.logger.log("pos ratio: {} neg ratio: {}".format(self.pos_ratio, self.neg_ratio))
            
            if self.config.mode == "train":
                self.train_dataset = CategoryDataset(self.root, "train", self.catid, "/mnt/scannet/tables", 
                                                     self.pos_ratio, self.neg_ratio, self.voxel_size)
            if self.config.mode in ["train", "val"]: 
                self.val_dataset = CategoryDataset(self.root, "val", self.catid, "/mnt/scannet/tables", 
                                                   self.pos_ratio, self.neg_ratio, self.voxel_size)
            if self.config.mode == "test":
                self.test_dataset = CategoryDataset(self.root, "test", self.catid, "/mnt/scannet/tables", 
                                                    self.pos_ratio, self.neg_ratio, self.voxel_size)
            
        else:
            raise ValueError("no dataset named {}".format(self.config.dataset))
        
        if self.config.mode == "train":
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, 
                                                            num_workers=4, collate_fn=self.train_dataset.collate_pair_fn)
        if self.config.mode in ["train", "val"]: 
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                          num_workers=2, collate_fn=self.val_dataset.collate_pair_fn)
        if self.config.mode == "test":
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                           num_workers=2, collate_fn=self.test_dataset.collate_pair_fn)
        
        
        self.logger.log("Finish Loading")
        
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
        elif self.config.embedding == "conv1_fc":
            assert len(self.config.dim) == 3
            conv_dim, linear1dim, linear2dim = self.config.dim
            self.embedding  = fc.conv1_fc_chamfer(conv_dim, linear1dim, linear2dim).to(self.device)
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

        
        params = []
        if self.config.train_ret:
            params.append({'params': self.embedding.parameters()})
            
        if self.config.train_pose:
            params.append({'params': self.model.parameters()})

        # Optimizer initialize
        self.logger.log("Using optimizer {}".format(self.config.optimizer))
        
        if self.config.optimizer == "SGD":
            self.optimizer = getattr(torch.optim, self.config.optimizer)(
                params,
                lr=self.config.lr,
                momentum=self.config.momentum,
            weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                params,
                lr=self.config.lr)
        else:
            raise ValueError("Optimizer {} not defined".format(self.config.optimizer))
            
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.exp_gamma)

        # Load pretrained FCGF model
        if self.config.pretrain and not self.config.resume:
            self.logger.log("loading pretrained model from {}".format(self.config.pretrain))
            self.model.load_state_dict(torch.load(self.config.pretrain)["state_dict"])
        
        # Resume from previous checkpoint
        if self.config.resume:
            self.logger.log("loading checkpoint from {}".format(self.config.resume))
            checkpoint = torch.load(self.config.resume)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.embedding.load_state_dict(checkpoint["embedding_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"]
            #self.start_epoch += 1
        
        self.pos_weight = 2.0
        self.neg_weight = 1.0
        self.feat_weight = 1.0

        self.pos_thres = 0.1
        self.neg_thres = 1.5

        # ransac params
        self.knn = 5
        self.max_corr_dist = 0.20
            
        # Loss
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
    def train_batch(self, data):
        # base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to(self.device)
        base_input = ME.SparseTensor(data["base_feat"], data["base_coords"], device=self.device)
        # pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to(self.device)
        pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"], device=self.device)
        # neg_input = ME.SparseTensor(data["neg_feat"], data["neg_coords"]).to(self.device)
        neg_input = ME.SparseTensor(data["neg_feat"], data["neg_coords"], device=self.device)

        # Pose pairs
        PiP = data["PiP_pairs"].long().to(self.device)
        PiN = data["PiN_pairs"].long().to(self.device)
        NiN = data["NiN_pairs"].long().to(self.device)

        base_output, base_feat = self.model(base_input)
        pos_output, pos_feat = self.model(pos_input)
        neg_output, neg_feat = self.model(neg_input)
        
        # Retrieval part
        if self.config.train_ret:
            base_feat = self.embedding(base_feat)
            pos_feat = self.embedding(pos_feat)
            neg_feat = self.embedding(neg_feat)
            
            batch_size = len(data["base_idx"])

            feat_loss = torch.Tensor([0.0]).float().to(self.device)
            
            if self.distance == "l2":
                base_feat_norm = F.normalize(base_feat, dim=1) 
                pos_feat_norm = F.normalize(pos_feat, dim=1) 
                neg_feat_norm = F.normalize(neg_feat, dim=1) 
                feat_loss = self.triplet_loss(base_feat_norm, pos_feat_norm, neg_feat_norm)

            else:
                base_feat_norm = F.normalize(base_feat.F, dim=1) 
                pos_feat_norm = F.normalize(pos_feat.F, dim=1) 
                neg_feat_norm = F.normalize(neg_feat.F, dim=1) 
                # Split the batch
                glob_base_feats = [];glob_pos_feats = [];glob_neg_feats = []
                for i in range(batch_size):
                    base_mask = base_feat.C[:,0]==i
                    pos_mask = pos_feat.C[:,0]==i
                    neg_mask = neg_feat.C[:,0]==i

                    glob_base_feats.append(base_feat_norm[base_mask, :])
                    glob_pos_feats.append(pos_feat_norm[pos_mask, :])
                    glob_neg_feats.append(neg_feat_norm[neg_mask, :])

                for i in range(batch_size):
                    pos_dist = self.chamfer_gpu(glob_base_feats[i], glob_pos_feats[i])
                    neg_dist = self.chamfer_gpu(glob_base_feats[i], glob_neg_feats[i])

                    feat_loss += F.relu(pos_dist - neg_dist + 1.0)

                feat_loss /= batch_size
        else:
            feat_loss = torch.Tensor([0.0]).float().to(self.device)

        # Pose part
        if self.config.train_pose:
            base_output = base_output.F
            pos_output = pos_output.F
            neg_output = neg_output.F

            BasePiP = base_output.index_select(0, PiP[:, 0])
            PosPiP =  pos_output.index_select(0, PiP[:, 1])
            BasePiN = base_output.index_select(0, PiN[:, 0])
            PosPiN = pos_output.index_select(0, PiN[:, 1])
            BaseNiN = base_output.index_select(0, NiN[:, 0])
            NegNiN = neg_output.index_select(0, NiN[:, 1])

            pos_loss = F.relu( (BasePiP - PosPiP).pow(2).sum(1).sqrt() - self.pos_thres).pow(2).mean()
            neg_loss0 = F.relu( self.neg_thres - (BasePiN - PosPiN).pow(2).sum(1).sqrt()).pow(2).mean()
            neg_loss1 = F.relu( self.neg_thres - (BaseNiN - NegNiN).pow(2).sum(1).sqrt()).pow(2).mean()
        else:
            pos_loss = torch.Tensor([0.0]).float().to(self.device)
            neg_loss0 = torch.Tensor([0.0]).float().to(self.device)
            neg_loss1 = torch.Tensor([0.0]).float().to(self.device)

        return pos_loss, neg_loss0, neg_loss1, feat_loss
    
    
    def train(self):
        self.logger.log("Start training")
        for epoch in range(self.start_epoch, self.max_epoch):
            lr = self.scheduler.get_last_lr()

            if (epoch+1) % self.eval_iter == 0:
                self.logger.log("Start evaluation")
                self.eval(mode="val", pose=self.config.train_pose, symmetry=self.config.use_symmetry)

            if self.config.train_ret:
                self.embedding.train()
            else:
                self.embedding.eval()

            if self.config.train_pose:
                self.model.train()
            else:
                self.model.eval()
                
            
            self.logger.log("Saving checkpoint")
            save_checkpoint(self.model, self.embedding, self.optimizer, self.scheduler, epoch, "./ckpts/", self.save_name)
            self.logger.log("save to {}".format(self.save_name))

            avg_pos_loss = 0;avg_neg_loss = 0;avg_feat_loss = 0
            
            for idx, data in enumerate(self.train_loader):
                
                base_idx, pos_idx, neg_idx = data["base_idx"], data["pos_idx"], data["neg_idx"]
                triplets = torch.stack([base_idx, pos_idx, neg_idx], 1)
                
                batch_size = len(data["base_idx"])
                
                pos_loss, neg_loss0, neg_loss1, feat_loss = self.train_batch(data)

                loss = self.pos_weight*pos_loss + self.neg_weight*(neg_loss0 + neg_loss1) + self.feat_weight*feat_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_pos_loss += self.pos_weight*pos_loss.item()*batch_size/len(self.train_dataset)
                avg_neg_loss += self.neg_weight*(neg_loss0.item()+neg_loss1.item())*batch_size/len(self.train_dataset)
                avg_feat_loss += self.feat_weight*feat_loss.item()*batch_size/len(self.train_dataset)
                

                if (idx+1) % self.log_batch == 0:
                    self.logger.log("EPOCH:{} Batch index: {}, Batch loss: {:.4f}".format(epoch+1, idx+1, loss.item()))
                    self.logger.log("loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f}".format(
                                    self.pos_weight*pos_loss.item(), 
                                    self.neg_weight*(neg_loss0.item()+neg_loss1.item()), 
                                    self.feat_weight*feat_loss.item()))

                torch.cuda.empty_cache()
                
            self.logger.log("EPOCH:{} SUMMARY loss: {:.4f} loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f}".format(
                            epoch+1, avg_pos_loss+avg_neg_loss+avg_feat_loss, avg_pos_loss, avg_neg_loss, avg_feat_loss))
            self.scheduler.step()
        
    def chamfer_gpu(self, pc0, pc1):
        
        pc0 = pc0[None,:,:]
        pc1 = pc1[:,None,:]
        delta = pc0 - pc1
        return delta.norm(dim=2).min(0)[0].mean() + delta.norm(dim=2).min(1)[0].mean()


    def eval(self, mode="val", pose=True, symmetry=True):
        print("start eval")
        np.random.seed(31)
        torch.random.manual_seed(31)
        self.model.eval()
        self.embedding.eval()

        base_outputs = [];pos_outputs = []
        base_origins = [];pos_origins = []

        glob_feats = []

        base_Ts = [];pos_Ts = [];
        base_syms = [];pos_syms = []
        
        if mode == "val":
            dataset = self.val_dataset
            loader = self.val_loader
        elif mode == "test":
            dataset = self.test_dataset
            loader = self.test_loader
        else:
            raise ValueError("No such mode: {}".format(mode))
        
        num_data = len(dataset)
        
        with torch.no_grad():

            for idx, data in enumerate(loader):
                print("Eval feature Index: {}/{}".format(idx+1, len(loader)))

                # base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")
                base_input = ME.SparseTensor(data["base_feat"], data["base_coords"], device=self.device)
                pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"], device=self.device)
                
                base_Ts.append(data["base_T"])
                pos_Ts.append(data["pos_T"])

                base_syms.append(data["base_sym"])
                pos_syms.append(data["pos_sym"])

                batch_size = len(data["base_idx"])

                base_output, base_feat = self.model(base_input)

                pos_output, pos_feat = self.model(pos_input)

                base_feat = self.embedding(base_feat)

                for i in range(batch_size):
                    base_mask = base_output.C[:,0]==i
                    pos_mask = pos_output.C[:,0]==i

                    base_outputs.append(base_output.F[base_mask, :])
                    pos_outputs.append(pos_output.F[pos_mask, :])
                    # print(data["base_origin"].device) # cpu
                    # device = base_mask.device
                    # data["base_origin"].to(device)
                    # print(base_mask.device) # cuda:0
                    # print(data["pos_origin"].device) # cpu
                    # data["pos_origin"].to(device)
                    # print(pos_mask.device) # cuda:0
                    
                    # print(f"base_mask shape: {base_mask.shape}")
                    # print(f"pos_mask shape: {pos_mask.shape}")
                    # print(f"data[base_origin] shape: {data['base_origin'].shape}")
                    # print(f"data[pos_origin] shape: {data['pos_origin'].shape}")

                    device1 = data["base_origin"].device
                    base_mask = base_mask.to(device1)
                    device2 = data["pos_origin"].device
                    pos_mask = pos_mask.to(device2)

                    # print(f"base_mask shape: {base_mask.shape}")
                    # print(f"pos_mask shape: {pos_mask.shape}")
                    # print(f"data[base_origin] shape: {data['base_origin'].shape}")
                    # print(f"data[pos_origin] shape: {data['pos_origin'].shape}")
                    
                    base_origins.append(data["base_origin"][base_mask, :])
                    pos_origins.append(data["pos_origin"][pos_mask, :])


                if self.distance == "l2":
                    base_feat_norm = F.normalize(base_feat, dim=1) #base_feat/base_feat.norm(dim=1, keepdim=True)
                    glob_feats.append(base_feat_norm)

                else:
                    base_feat_norm = F.normalize(base_feat.F, dim=1) #base_feat/base_feat.norm(dim=1, keepdim=True)

                    for i in range(batch_size):
                        feat_mask = base_feat.C[:,0]==i
                        glob_feats.append(base_feat_norm[feat_mask, :])  


            base_Ts = torch.cat(base_Ts, 0)
            pos_Ts = torch.cat(pos_Ts, 0)
            base_syms = torch.cat(base_syms, 0)
            pos_syms = torch.cat(pos_syms, 0)

            self.logger.log("-----TEST-----")
            if self.distance == "chamfer":
                dist = np.zeros([len(glob_feats), len(glob_feats)])
                for i in range(len(glob_feats)):
                    for j in range(i+1, len(glob_feats)):
                        dist[i,j] = self.chamfer_gpu(glob_feats[i], glob_feats[j])
                dist += dist.T

                stat = retrieval_dist(dist, dataset.pos_ratio, dataset.dist_mat)
                self.logger.log("mAP: {} percision: {} top1_error: {}".format(stat["mAP"], stat["percision"], stat["top1_error"]))
            else:
                descriptor = torch.cat(glob_feats, 0).detach().cpu().numpy()
                stat = retrieval_eval(descriptor, dataset.pos_ratio, dataset.dist_mat)
                self.logger.log("mAP: {} percision: {} top1_error: {}".format(stat["mAP"], stat["percision"], stat["top1_error"]))

            if pose:

                t_losses = []
                r_losses = []

                for idx in range(len(dataset)):
                    if (idx+1) % 10 == 0:
                        t_losses_avg = sum(t_losses)/len(t_losses)
                        r_losses_avg = sum(r_losses)/len(r_losses)
                        self.logger.log("Eval align Index: {}/{} T error: {:.4f} R error: {:.4f} ".format(idx+1, len(dataset), 
                                        t_losses_avg, r_losses_avg))

                    # matching based pose estimation
                    xyz0, xyz1, T0, T1 = base_origins[idx], pos_origins[idx], base_Ts[idx,:,:], pos_Ts[idx,:,:]
                    baseF, posF = base_outputs[idx], pos_outputs[idx]

                    base_sym = base_syms[idx].item()
                    pos_sym = pos_syms[idx].item()

                    # RANSAC
                    idx_0, idx_1 = find_kcorr(baseF, posF, k=self.knn, subsample_size=5000)

                    source_pcd = xyz0[idx_0]
                    target_pcd = xyz1[idx_1]

                    T_est = registration_based_on_corr(source_pcd, target_pcd, self.max_corr_dist)

                    T_est = torch.from_numpy(T_est.astype(np.float32))

                    t_loss, r_loss = eval_pose(T_est, T0, T1, axis_symmetry=max(base_sym, pos_sym))
                                        
                    t_losses.append(t_loss)
                    r_losses.append(r_loss)

                t_losses_avg = sum(t_losses)/len(t_losses)
                r_losses_avg = sum(r_losses)/len(r_losses)
                self.logger.log("T error: {}, R error: {}".format(t_losses_avg, r_losses_avg))
