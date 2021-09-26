import torch
import pytorch_lightning as pl
import os

from deeplearning import CNNSegmentation

model_input_path = os.path.join("lightning_logs/rgb_001/checkpoints/epoch=140-step=1691.ckpt")
model_output_path = os.path.join("model/model_jit.pt")

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.model = CNNSegmentation.UNet(n_class=2, conv_block=["VGG", "VGG", "VGG"], in_channel_dim=3)
    
    def forward(self, x):
        return self.model(x)

model = SegmentationModel()
model = model.load_from_checkpoint(model_input_path)
script = model.to_torchscript()
torch.jit.save(script, model_output_path)
