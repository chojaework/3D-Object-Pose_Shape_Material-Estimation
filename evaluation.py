from config import get_config
from trainer.RetrievalTrainer import RetrievalTrainer
from trainer.ScannetTrainer import ScannetTrainer
from trainer.ResnetTrainer import ResnetTrainer


if __name__ == "__main__": 
    config = get_config()
    if config.trainer == "RetrievalTrainer":
        print(config.use_symmetry)
        trainer = RetrievalTrainer()
        trainer.eval(config.mode, pose=False, symmetry=config.use_symmetry)
    elif config.trainer == "ScannetTrainer":
        trainer = ScannetTrainer()
        trainer.eval(config.mode, pose=False, symmetry=config.use_symmetry)
    elif config.trainer == "ResNetTrainer":
        trainer = ResnetTrainer()
        trainer.eval("test")
    else:
        raise ValueError("Unknown trainer {}".format(config.trainer))
        
