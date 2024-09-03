import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
from datasets.ShapenetDataset import *
from datasets.CategoryDataset import *

from trainer.RetrievalTrainer import *

from utils.logger import logger
from utils.ckpts import load_checkpoint, save_checkpoint
from utils.retrieval import compute_mAP, retrieval_eval
from model import load_model, fc
from config import get_config

import MinkowskiEngine as ME

class RetrievalPlusTrainer(RetrievalTrainer):
    def __init__(self):
        RetrievalTrainer.__init__(self)
        
        self.center_weight = 1
        
    def train_batch(self, data):
        # Self rotation
        base1_input = ME.SparseTensor(feats=data["base_feat"], coords=data["base_coords"]).to(self.device)
        base2_input = ME.SparseTensor(feats=data["base2_feat"], coords=data["base2_coords"]).to(self.device)
        pos_input = ME.SparseTensor(data["pos_feat"], data["pos_coords"]).to(self.device)
        neg_input = ME.SparseTensor(data["neg_feat"], data["neg_coords"]).to(self.device)

        PiP = data["PiP_pairs"].long().to(self.device)
        PiN = data["PiN_pairs"].long().to(self.device)
        NiN = data["NiN_pairs"].long().to(self.device)

        base1_output, base1_feat = self.model(base1_input)
        base2_output, base2_feat = self.model(base2_input)
        pos_output, pos_feat = self.model(pos_input)
        neg_output, neg_feat = self.model(neg_input)

        base1_feat, base1_prob = self.embedding(base1_feat)
        base2_feat, base2_prob = self.embedding(base2_feat)
        pos_feat, pos_prob = self.embedding(pos_feat)
        neg_feat, neg_prob = self.embedding(neg_feat)

        
        
        base1_norm = base1_feat/base1_feat.norm(dim=1, keepdim=True)
        base2_norm = base2_feat/base2_feat.norm(dim=1, keepdim=True)
        pos_norm = pos_feat/pos_feat.norm(dim=1, keepdim=True)
        neg_norm = neg_feat/neg_feat.norm(dim=1, keepdim=True)

        center_loss = (base1_norm-base2_norm).pow(2).sum(1).mean()
        center_feat = base1_norm + base2_norm
        center_norm = center_feat/center_feat.norm(dim=1, keepdim=True)

        feat_loss = self.triplet_loss(center_norm, pos_norm, neg_norm)
            
        # Pose Loss 
        if self.config.train_pose:
            base_output = base1_output.F
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
            pos_loss = torch.tensor([0]).to(self.device)
            neg_loss0 = torch.tensor([0]).to(self.device)
            neg_loss1 = torch.tensor([0]).to(self.device)
            
        return pos_loss, neg_loss0, neg_loss1, feat_loss, center_loss
    
    
    def train(self, ):
        self.logger.log("Start training")
        for epoch in range(self.start_epoch, self.max_epoch):
            lr = self.scheduler.get_last_lr()

            if (epoch+1) % self.eval_iter == 0:
                self.logger.log("Start evaluation")
                if self.config.dataset in ["CategoryDataset"]:
                    self.eval("chamfer", False)
                else:
                    self.eval("subcategory", False)
            
            self.logger.log("Saving checkpoint")
            save_checkpoint(self.model, self.embedding, self.optimizer, self.scheduler, epoch, "./ckpts/", self.save_name)
            self.logger.log("save to {}".format(self.save_name))

            avg_pos_loss = 0;avg_neg_loss = 0;avg_feat_loss = 0;avg_center_loss = 0
            
            for idx, data in enumerate(self.train_loader):
                #self.logger.log("start")
                
                batch_size = len(data["base_idx"])
                
                pos_loss, neg_loss0, neg_loss1, feat_loss, center_loss = self.train_batch(data)
                
                loss = self.pos_weight*pos_loss + self.neg_weight*(neg_loss0 + neg_loss1) + \
                       self.feat_weight*feat_loss + 1*center_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_pos_loss += self.pos_weight*pos_loss.item()*batch_size/len(self.train_dataset)
                avg_neg_loss += self.neg_weight*(neg_loss0.item()+neg_loss1.item())*batch_size/len(self.train_dataset)
                avg_feat_loss += self.feat_weight*feat_loss.item()*batch_size/len(self.train_dataset)
                avg_center_loss += self.center_weight*center_loss.item()*batch_size/len(self.train_dataset)


                if (idx+1) % self.log_batch == 0:
                    self.logger.log("EPOCH:{} Batch index: {}, Batch loss: {:.4f}".format(epoch+1, idx+1, loss.item()))
                    self.logger.log("loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f} loss cent: {:.4f}".format(
                                    self.pos_weight*pos_loss.item(), 
                                    self.neg_weight*(neg_loss0.item()+neg_loss1.item()), 
                                    self.feat_weight*feat_loss.item(),
                                    self.center_weight*center_loss.item()
                                   ))
                    

                torch.cuda.empty_cache()
                #self.logger.log("end")
            avg_loss = avg_pos_loss+avg_neg_loss+avg_feat_loss
            self.logger.log("EPOCH:{} SUMMARY loss: {:.4f} loss pos: {:.4f} loss neg: {:.4f} loss feat: {:.4f} loss cent: {:.4f}".format(
                epoch+1, avg_loss, avg_pos_loss, avg_neg_loss, avg_feat_loss, avg_center_loss))
            self.scheduler.step()
