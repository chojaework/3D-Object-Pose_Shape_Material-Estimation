import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from datasets.ChairDataset import *
from datasets.CategoryDataset import *
from datasets.HardCategoryDataset import *

from trainer.RetrievalTrainer import *
from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import compute_mAP, retrieval_eval
from model import load_model, fc, resnet
from config import get_config

import MinkowskiEngine as ME


class ResnetTrainer(trainer):

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
                self.train_dataset = CategoryDataset(self.root, "train", self.catid, "/scannet/tables", 
                                                     self.pos_ratio, self.neg_ratio, self.voxel_size)
            if self.config.mode in ["train", "val"]: 
                self.val_dataset = CategoryDataset(self.root, "val", self.catid, "/scannet/tables", 
                                                   self.pos_ratio, self.neg_ratio, self.voxel_size)
            if self.config.mode in ["train", "test"]:
                self.test_dataset = CategoryDataset(self.root, "test", self.catid, "/scannet/tables", 
                                                    self.pos_ratio, self.neg_ratio, self.voxel_size)
            
        else:
            raise ValueError("no dataset named {}".format(self.config.dataset))
        
        if self.config.mode == "train":
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, 
                                                            num_workers=4, collate_fn=self.train_dataset.collate_pair_fn)
        if self.config.mode in ["train", "val"]: 
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                          num_workers=2, collate_fn=self.val_dataset.collate_pair_fn)
        if self.config.mode in ["train", "test"]:
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, 
                                                           num_workers=2, collate_fn=self.test_dataset.collate_pair_fn)


        self.logger.log("Finish Loading")
        
        # Model initialize
        # ResNet18
        model = resnet.ResNet18(1, 256)
        self.model = model.to(self.device)
        # Embedding network for retrieval
    
        # optimizer initialize
        self.logger.log("Using optimizer {}".format(self.config.optimizer))
        
        if self.config.optimizer == "SGD":
            self.optimizer = getattr(torch.optim, self.config.optimizer)(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr)
        else:
            raise ValueError("no optimizer {}".format(self.config.optimizer))

        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.exp_gamma)


        # Load pretrained FCGF model
        if self.config.pretrain:
            self.logger.log("loading pretrained model from {}".format(self.config.pretrain))
            self.model.load_state_dict(torch.load(self.config.pretrain)["state_dict"])
        
        # Resume from previous checkpoint
        if self.config.resume:
            self.logger.log("loading checkpoint from {}".format(self.config.resume))
            checkpoint = torch.load(self.config.resume)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.start_epoch = checkpoint["epoch"]
            #self.start_epoch += 1
        

        self.feat_weight = 1.0
        self.center_weight = 1.0
            
        # Loss
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
    def train_batch(self, data):
        batch_size = len(data["base_idx"])

        
        base1_input = ME.SparseTensor(feats=data["base_feat"], coords=data["base_coords"]).to(self.device)
        pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to(self.device)
        neg_input = ME.SparseTensor(data["neg_feat"], data["neg_coords"]).to(self.device)

        base1_feat = self.model(base1_input).F
        pos_feat = self.model(pos_input).F
        neg_feat = self.model(neg_input).F
                
        base1_norm = F.normalize(base1_feat, dim=1)
        pos_norm = F.normalize(pos_feat, dim=1)
        neg_norm = F.normalize(neg_feat, dim=1)
        
        feat_loss = self.triplet_loss(base1_norm, pos_norm, neg_norm)
        
        return feat_loss
    
    
    def train(self, ):
        self.logger.log("Start training")
        for epoch in range(self.start_epoch, self.max_epoch):
            lr = self.scheduler.get_last_lr()
            self.logger.log("lr: {}".format(lr))
            if (epoch+1) % self.eval_iter == 0:
                self.logger.log("Start evaluation")
                self.eval("val")

            
            self.logger.log("Saving checkpoint")
            save_checkpoint(self.model, None, self.optimizer, self.scheduler, epoch, "./ckpts/", self.save_name)
            self.logger.log("save to {}".format(self.save_name))

            avg_feat_loss = 0;avg_center_loss = 0
            
            for idx, data in enumerate(self.train_loader):
                #self.logger.log("start")
                
                batch_size = len(data["base_idx"])
                
                feat_loss = self.train_batch(data)
                
                #print(feat_loss)
                #print(center_loss)
                
                loss = self.feat_weight*feat_loss #+ self.center_weight*center_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_feat_loss += self.feat_weight*feat_loss.item()*batch_size/len(self.train_dataset)

                if (idx+1) % self.log_batch == 0:
                    self.logger.log("EPOCH:{} Batch index: {}, Batch loss: {:.4f}".format(epoch+1, idx+1, loss.item()))

                torch.cuda.empty_cache()
                #self.logger.log("end")
            avg_loss = avg_feat_loss
            self.logger.log("EPOCH:{} SUMMARY loss: {:.4f} loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f}".format(
                epoch+1, avg_loss, 0, 0, avg_feat_loss))
            self.scheduler.step()

            
    def eval(self, mode):
        """
        batch size have to be 1
        """
        self.model.eval()

        if mode == "val":
            dataset = self.val_dataset
            loader = self.val_loader
        elif mode == "test":
            dataset = self.test_dataset
            loader = self.test_loader
        else:
            raise ValueError("no dataset")

        glob_feats = []

        num_data = len(dataset)
        for idx, data in enumerate(loader):
            self.logger.log("Eval feature Index: {}/{}".format(idx+1, len(loader)))
            
            with torch.no_grad():
                base_input = ME.SparseTensor(data["base_feat"], data["base_coords"]).to(self.device)
                                
                base_feat = self.model(base_input).F
                
                # global feature
                base_norm = F.normalize(base_feat, dim=1)
                
                glob_feats.append(base_norm)

        
        self.logger.log("-----TEST-----")
        # for all chair, evaluate retrieval with chamfer distance rank
        glob_feats = torch.cat(glob_feats, 0).detach().cpu().numpy()
        
        stat = retrieval_eval(glob_feats, dataset.pos_ratio, dataset.table)

        self.logger.log("mAP: {} percision: {} top1_error: {}".format(stat["mAP"], stat["percision"], stat["top1_error"]))

        self.model.train()
