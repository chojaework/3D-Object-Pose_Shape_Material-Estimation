from config import get_config
import numpy as np
from trainer.RetrievalTrainer import RetrievalTrainer
from trainer.ScannetTrainer import ScannetTrainer
from trainer.ResnetTrainer import ResnetTrainer

np.random.seed(31)


if __name__ == "__main__": 
    config = get_config()
    if config.trainer == "RetrievalTrainer":
        trainer = RetrievalTrainer()
        trainer.train()
    elif config.trainer == "ScannetTrainer":
        trainer = ScannetTrainer()
        trainer.train()
        trainer.eval("val", pose=config.train_pose, symmetry=config.use_symmetry)
        trainer.eval("test", pose=config.train_pose, symmetry=config.use_symmetry)
    elif config.trainer == "ResNetTrainer":
        trainer = ResnetTrainer()
        trainer.train()
        trainer.eval("test")

    else:
        raise ValueError("Unknown trainer {}".format(config.trainer))
