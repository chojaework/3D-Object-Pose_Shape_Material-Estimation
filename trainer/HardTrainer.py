import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from datasets.ChairDataset import *
from datasets.CategoryDataset import *
from datasets.HardCategoryDataset import *

from trainer.RetrievalTrainer import trainer

from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import *
from model import load_model, fc
from config import get_config

import MinkowskiEngine as ME


class HardTrainer(trainer):

    def __init__(self, ):
        trainer.__init__(self)

        # dataset and dataloader
        if self.config.dataset == "HardCategoryDataset":
            self.pos_ratio = self.config.pos_ratio
            self.neg_ratio = self.config.neg_ratio
            self.catid = self.config.catid
            self.logger.log("CatId: {}".format(self.catid))
            self.logger.log("pos ratio: {} neg ratio: {}".format(self.pos_ratio, self.neg_ratio))
            self.train_dataset = HardCategoryDataset(self.root, "train", self.catid, "/scannet/tables", 
                                                 self.pos_ratio, self.neg_ratio, self.voxel_size)
            self.val_dataset = HardCategoryDataset(self.root, "val", self.catid, "/scannet/tables", 
                                               self.pos_ratio, self.neg_ratio, self.voxel_size)
            self.test_dataset = HardCategoryDataset(self.root, "test", self.catid, "/scannet/tables", 
                                                self.pos_ratio, self.neg_ratio, self.voxel_size)
            
        else:
            raise ValueError("no dataset named {}".format(self.config.dataset))

            
        self.logger.log("Train pos range: {} neg range: {} ".format(self.train_dataset.pos_n, self.train_dataset.neg_n))
        self.logger.log("Test pos range: {} neg range: {} ".format(self.test_dataset.pos_n, self.test_dataset.neg_n))
        self.logger.log("Val pos range: {} neg range: {} ".format(self.val_dataset.pos_n, self.val_dataset.neg_n))

        self.logger.log("Train dataset size: {} Test dataset size: {} Val dataset size: {}".format(
            len(self.train_dataset), len(self.test_dataset), len(self.val_dataset)))

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, 
                                                        num_workers=4, collate_fn=self.train_dataset.collate_pair_fn)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                        num_workers=2, collate_fn=self.test_dataset.collate_pair_fn)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                            num_workers=2, collate_fn=self.val_dataset.collate_pair_fn)


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
        if self.config.mode == "chamfer":
            self.embedding = fc.conv1_chamfer(256).to(self.device)
        elif self.config.mode == "mean":
            self.embedding = fc.avg_embedding(256, 256).to(self.device)
        
        params = []
        if self.config.train_ret:
            params.append({'params': self.embedding.parameters()})
            
        if self.config.train_pose:
            params.append({'params': self.model.parameters()})

        # optimizer initialize
        self.logger.log("Using optimizer {}".format(self.config.optimizer))
        
        
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            params,
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay)
        """
        self.optimizer = optim.Adam(
            [{'params': self.model.parameters()}, {'params': self.embedding.parameters()}],
            lr=self.config.lr)"""

        
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
        self.cls_weight = 1.0

        self.pos_thres = 0.1
        self.neg_thres = 1.5
            
        # Loss
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
       
    def split_batch(self, sp, feat, batch_size):
        glob_feats = []
        for i in range(batch_size):
            base_mask = sp.C[:,0]==i
            glob_feats.append(feat[base_mask, :])

        return glob_feats
    
    def mean_loss(self, base_feat, pos_feat, neg_feat, data, batch_size):
        
        glob_base_feats = base_feat/base_feat.norm(dim=1, keepdim=True)
        glob_pos_feats = pos_feat/pos_feat.norm(dim=1, keepdim=True)
        glob_neg_feats = neg_feat/neg_feat.norm(dim=1, keepdim=True)
        
        glob_base_feats = glob_base_feats.view(batch_size, 1, -1)
        glob_pos_feats = glob_pos_feats.view(batch_size, 1, -1)
        glob_neg_feats = glob_neg_feats.view(batch_size, 4, -1)

        far = 0
        
        pos_dist = (glob_base_feats[:, 0, :] - glob_pos_feats[:, far, :]).norm(dim=1, keepdim=True)
        neg_dist = (glob_base_feats - glob_neg_feats).norm(dim=2)
        
        feat_loss = (F.relu(pos_dist - neg_dist + 1.0)).mean()
        
        
        return feat_loss

    
    def chamfer_loss(self, base_feat, pos_feat, neg_feat, data, batch_size):
        
        base_feat_norm = base_feat.F/base_feat.F.norm(dim=1, keepdim=True)
        pos_feat_norm = pos_feat.F/pos_feat.F.norm(dim=1, keepdim=True)
        neg_feat_norm = neg_feat.F/neg_feat.F.norm(dim=1, keepdim=True)

        feat_loss = torch.tensor([0.0]).float().to(self.device)

        glob_base_feats = self.split_batch(base_feat, base_feat_norm, len(data["base_idx"]))
        glob_pos_feats = self.split_batch(pos_feat, pos_feat_norm, len(data["pos_idx"]))
        glob_neg_feats = self.split_batch(neg_feat, neg_feat_norm, len(data["neg_idx"]))

        for i in range(batch_size):
            pos_dist = self.chamfer_gpu(glob_base_feats[i], glob_pos_feats[i])
            neg_dist0 = self.chamfer_gpu(glob_base_feats[i], glob_neg_feats[i*4])
            neg_dist1 = self.chamfer_gpu(glob_base_feats[i], glob_neg_feats[i*4+1])
            neg_dist2 = self.chamfer_gpu(glob_base_feats[i], glob_neg_feats[i*4+2])
            neg_dist3 = self.chamfer_gpu(glob_base_feats[i], glob_neg_feats[i*4+3])

            feat_loss += F.relu(pos_dist - neg_dist0 + 1.0)/4
            feat_loss += F.relu(pos_dist - neg_dist1 + 1.0)/4
            feat_loss += F.relu(pos_dist - neg_dist2 + 1.0)/4
            feat_loss += F.relu(pos_dist - neg_dist3 + 1.0)/4

        feat_loss /= batch_size
        
        return feat_loss
            
    
    def train_batch(self, data):
        base_input = ME.SparseTensor(feats=data["base_feat"], coords=data["base_coords"]).to(self.device)
        pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to(self.device)
        neg_input = ME.SparseTensor(data["neg_feat"], data["neg_coords"]).to(self.device)

        # Pose pairs
        #PiP = data["PiP_pairs"].long().to(self.device)
        #PiN = data["PiN_pairs"].long().to(self.device)
        #NiN = data["NiN_pairs"].long().to(self.device)

        base_output, base_feat = self.model(base_input)
        pos_output, pos_feat = self.model(pos_input)
        neg_output, neg_feat = self.model(neg_input)
        
        if self.config.train_ret:
            base_feat = self.embedding(base_feat)
            pos_feat = self.embedding(pos_feat)
            neg_feat = self.embedding(neg_feat)
            
            batch_size = len(data["base_idx"])
            
            if self.config.mode == "chamfer":
                feat_loss = self.chamfer_loss(base_feat, pos_feat, neg_feat, data, batch_size)
            elif self.config.mode == "mean":
                feat_loss = self.mean_loss(base_feat, pos_feat, neg_feat, data, batch_size)

        # pose part
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
            pos_loss = torch.tensor([0.0]).float().to(self.device)
            neg_loss0 = torch.tensor([0.0]).float().to(self.device)
            neg_loss1 = torch.tensor([0.0]).float().to(self.device)

        return pos_loss, neg_loss0, neg_loss1, feat_loss
    
    
    def train(self, ):
        self.logger.log("Start training")
        for epoch in range(self.start_epoch, self.max_epoch):
            lr = self.scheduler.get_last_lr()

            if (epoch+1) % self.eval_iter == 0:
                self.logger.log("Start evaluation")
                self.eval(self.config.mode, self.config.train_pose)

            
            self.logger.log("Saving checkpoint")
            save_checkpoint(self.model, self.embedding, self.optimizer, self.scheduler, epoch, "./ckpts/", self.save_name)
            self.logger.log("save to {}".format(self.save_name))

            avg_pos_loss = 0;avg_neg_loss = 0;avg_feat_loss = 0
            
            for idx, data in enumerate(self.train_loader):
                #self.logger.log("start")
                
                base_idx, pos_idx, neg_idx = data["base_idx"], data["pos_idx"], data["neg_idx"]
                
                batch_size = len(data["base_idx"])
                
                pos_loss, neg_loss0, neg_loss1, feat_loss = self.train_batch(data)
                #print(pos_loss, neg_loss0, neg_loss1, feat_loss)
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
                    
                    #if self.config.train_cls:
                    #    self.logger.log("loss cls: {}".format(cls_weight*cls_loss.item()))

                torch.cuda.empty_cache()
                #self.logger.log("end")
                
            self.logger.log("EPOCH:{} SUMMARY loss: {:.4f} loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f}".format(
                            epoch+1, avg_pos_loss+avg_neg_loss+avg_feat_loss, avg_pos_loss, avg_neg_loss, avg_feat_loss))
            self.scheduler.step()
        
    def chamfer_gpu(self, pc0, pc1):
        pc0 = pc0[None,:,:]
        pc1 = pc1[:,None,:]
        delta = pc0 - pc1
        # print((pc0-pc1).shape)
        #return np.mean(np.min(np.linalg.norm(delta, 2, 2), 1)) + np.mean(np.min(np.linalg.norm(delta, 2, 2), 0))
        return delta.norm(dim=2).min(0)[0].mean() + delta.norm(dim=2).min(1)[0].mean()


    def eval(self, mode="chamfer", pose=False):
        """
        mode is [chamfer, mean]
        """
        self.model.eval()
        self.embedding.eval()
        
        glob_feats = []

        num_data = len(self.val_dataset)
        for idx, data in enumerate(self.val_loader):
            print("Eval feature Index: {}/{}".format(idx+1, len(self.val_loader)))

            with torch.no_grad():
                base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to("cuda")
                pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to("cuda")

                batch_size = len(data["base_idx"])
                
                base_output, base_feat = self.model(base_input)
                base_feat = self.embedding(base_feat)

                #pos_output, pos_feat = self.model(pos_input)

                if mode == "chamfer":
                    
                    base_feat_norm = base_feat.F/base_feat.F.norm(dim=1, keepdim=True)

                    for i in range(batch_size):
                        feat_mask = base_feat.C[:,0]==i
                        glob_feats.append(base_feat_norm[feat_mask, :]) 
                        
                elif mode == "mean":
                    
                    base_feat_norm = base_feat/base_feat.norm(dim=1, keepdim=True)
                    glob_feats.append(base_feat_norm)
                    

        self.logger.log("-----TEST-----")

        if mode == "chamfer":

            dist = np.zeros([len(glob_feats), len(glob_feats)])
            for i in range(len(glob_feats)):
                for j in range(i+1, len(glob_feats)):
                    dist[i,j] = self.chamfer_gpu(glob_feats[i], glob_feats[j])
            dist += dist.T

            stat = retrieval_dist(dist, self.val_dataset.pos_ratio, self.val_dataset.table)
        
        elif mode == "mean":
            glob_feats = torch.cat(glob_feats, 0).cpu().numpy()
            stat = retrieval_eval(glob_feats, self.val_dataset.pos_ratio, self.val_dataset.table)

        self.logger.log("mAP: {} percision: {}".format(stat["mAP"], stat["percision"]))


        self.model.train()
        self.embedding.train()

            
            
            
if __name__ == "__main__": 
    config = get_config()
    if config.trainer == "RetrievalTrainer":
        trainer = RetrievalTrainer()
    elif config.trainer == "HardTripletTrainer":
        trainer = HardTripletTrainer()
    else:
        raise ValueError("Unknown trainer {}".format(config.trainer))
    trainer.train()
