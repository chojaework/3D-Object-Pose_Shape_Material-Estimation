import os
import numpy as np
import torch
from utils.read_json import load_csv
from utils.preprocess import path_dict

from test_time.CADFeatureCollection import CADFeatureCollection
from datasets.Reader import CategoryLibReader, Scan2cadLibReader

class RetrievalModule:
    """
    Retrieval Module contains extracted features of certain category in shapenet
    """
    def __init__(self, config, feature_extractor, dataset, update=False):
        
        self.config = config
        self.root = self.config.root
        self.catid = self.config.catid
        #self.lib_path = self.config.lib_path

        self.distance = feature_extractor.distance
        self.model = feature_extractor.model
        self.embedding = feature_extractor.embedding
        self.id2path = path_dict(self.root)

        if dataset == "shapenet":
            self.load_shapenet(update)
        elif dataset == "scan2cad":
            self.load_scan2cad(update)
        else:
            raise ValueError("Unknown dataset {}".format(dataset))


    def load_scan2cad(self, update):
        """
        Load CAD models used in Scan2CAD dataset
        """
        scan2cad_objs = load_csv(self.config.scan2cad_dict)

        ids = []
        for catId, objId in scan2cad_objs:
            if catId == self.catid:
                ids.append(objId)
        reader = Scan2cadLibReader(self.root, self.catid, ids, self.id2path, 1)

        #exist_lib = os.listdir(os.path.join(self.lib_path, self.catid))
        #if not "{}_{}.npy".format(self.catid, "scan2cad") in exist_lib or update: 
        extra = {"scan2cad_dict": self.config.scan2cad_dict}
        cat_collector = CADFeatureCollection(self.root, self.catid, "scan2cad", self.distance, 32, 0.03, self.model, self.embedding, extra)
        feats = cat_collector.collect()
        #np.save(os.path.join(self.lib_path, self.catid, "{}_{}.npy".format(self.catid, "scan2cad")), feats)

        self.pc_names = reader.files
        self.feat_lib = feats#np.load(os.path.join(self.lib_path, self.catid, "{}_{}.npy".format(self.catid, "scan2cad")))

    def TopN(self, test_feat, N=1):
        """
        retrieve top N similar CAD models
        """
        dist = np.linalg.norm(self.feat_lib - test_feat, axis=1)
        rank = np.argsort(dist)
        topn_idx = rank[:N]

        similar_pcs = [self.pc_names[i] for i in topn_idx]
        
        return similar_pcs


    """
    def load_shapenet(self, update):
        
        #Load subset of Shapenet
        
        available_split = ["train", "test", "val"]

        if not os.path.exists(self.lib_path):
            os.mkdir(self.lib_path)

        if not os.path.exists(os.path.join(self.lib_path, self.catid)):
            os.mkdir(os.path.join(self.lib_path, self.catid))

        exist_lib = os.listdir(os.path.join(self.lib_path, self.catid))
        for try_split in available_split:
            if not "{}_{}.npy".format(self.catid, try_split) in exist_lib or update: 
                cat_collector = CADFeatureCollection(self.root, self.catid, try_split, self.distance, 32, 0.03, self.model, self.embedding)
                feats = cat_collector.collect()
                np.save(os.path.join(self.lib_path, self.catid, "{}_{}.npy".format(self.catid, try_split)), feats)

        train_feats = np.load(os.path.join(self.lib_path, self.catid, "{}_train.npy".format(self.catid)))
        val_feats = np.load(os.path.join(self.lib_path, self.catid, "{}_val.npy".format(self.catid)))
        test_feats = np.load(os.path.join(self.lib_path, self.catid, "{}_test.npy".format(self.catid)))
        
        train_reader = CategoryLibReader(self.root, self.catid, ["train"], 1)
        val_reader = CategoryLibReader(self.root, self.catid, ["val"], 1)
        test_reader = CategoryLibReader(self.root, self.catid, ["test"], 1)

        self.pc_names = train_reader.files + val_reader.files + test_reader.files
        self.feat_lib = np.concatenate([train_feats, val_feats, test_feats], 0)
    """