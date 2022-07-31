
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
import importlib
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")    
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    mc = ModelCheckpoint(**cfg.model.model_checkpoint)
    trainer = Trainer(**cfg.model.trainer)
    logger = WandbLogger(**cfg.model.logger)
    model = importlib.import_module("lightning_models." + cfg.model.model_loc)
    # dataset = importlib.import_module("data_modules." + cfg.data.module_name)
if __name__ == "__main__":
    train()