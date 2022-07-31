from typing import Any
from pytorch_lightning import LightningDataModule, LightningModule

def get_model(args):
    return LightningModel(args)

class LightningModel(LightningModule):
    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
    
    def forward(self ,x):
        return x
    
    def training_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass