import cv2
import numpy as np
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import yaml
from sklearn.model_selection import KFold

from deeplearning import CNNSegmentation

config_path = 'config/config.yml'
#load configuration
with open(config_path, 'r') as g:
    config = yaml.load(g, Loader=yaml.FullLoader)

# Parameter Training
dataset_dir = config["dataset_path"]

# Hyperparameter
val_size = config["val_size"]
random_seed = config['random_seed']
epoch = config['epoch']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
in_channel = config['in_channel_dim']

# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, data_train, data_test, split="train", transform=None,in_channel=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.in_channel = in_channel
        self.file_list = file_list
        if split == "train":
            self.file_list = data_train
        elif split == "valid":
            self.file_list = data_test
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        # Load image
        if self.in_channel == 1:
            image = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image,axis=2)
        elif self.in_channel == 3:
            image = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
        # Load mask
        mask_0 = cv2.imread(os.path.join(self.dataset_dir, "masks", "bad", filename + ".png"), cv2.IMREAD_GRAYSCALE)
        mask_1 = cv2.imread(os.path.join(self.dataset_dir, "masks", "good", filename + ".png"), cv2.IMREAD_GRAYSCALE)
        if(mask_0 is not None and mask_1 is not None):
            mask = np.dstack((mask_0, mask_1))
        elif(mask_0 is not None):
            mask=np.dstack((mask_0,np.zeros((image.shape[0],image.shape[1]))))
        elif(mask_1 is not None):
            mask=np.dstack((np.zeros((image.shape[0],image.shape[1])),mask_1))
        else:
            mask = np.zeros((image.shape[0],image.shape[1], 2))
        # Apply Augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        # Normalization
        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255
        # Transpose
        image = image.transpose(2,0,1)
        mask = mask.transpose(2,0,1)

        return image, mask

# Loss Function
def BCEDiceLoss(Yp, Yt):
    #hitung BCE with logits loss
    bce = torch.nn.functional.binary_cross_entropy_with_logits(Yp, Yt.type_as(Yp))
    smooth = 1e-5 #toleransi smooth
    #hitung dice loss
    Yp = torch.sigmoid(Yp)
    num = Yt.size(0)
    #.view(-1) = flatten matrix tensor
    Yp = Yp.view(num, -1)
    Yt = Yt.view(num, -1)
    intersection = (Yp * Yt)
    dice = (2. * intersection.sum(1) + smooth) / (Yp.sum(1) + Yt.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    #loss calculation
    bce_dice_loss = 0.5 * bce + dice
    return bce_dice_loss

# Metric
#1. PERHITUNGAN IoU
def iou_score(Yp, Yt):
    smooth = 1e-5 #toleransi smooth
    #data sebelumnya telah dijadikan tensor untuk dibaca di GPU
    #dikembalikan ke CPU (numpy) dulu untuk perhitungan IoU
    Yp = torch.sigmoid(Yp).data.cpu().numpy()
    Yt = Yt.data.cpu().numpy()
    #threshold confidence 0.5 saja yang akan diproses
    output_ = Yp > 0.5
    target_ = Yt > 0.5
    intersection = (output_ & target_).sum() #irisan
    union = (output_ | target_).sum() #union
    #rumus IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou
    
#2. PERHITUNGAN DICE COEFFICIENT
def dice_coef(Yp, Yt):
    smooth = 1e-5 #toleransi smooth
    #data sebelumnya telah dijadikan tensor untuk dibaca di GPU
    #maka dikembalikan ke CPU (numpy) dulu untuk perhitungan IoU
    #.view(-1) = flatten matrix tensor
    Yp = torch.sigmoid(Yp).view(-1).data.cpu().numpy()
    Yt = Yt.view(-1).data.cpu().numpy()
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice = (2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth)
    return dice

# Model
## UNET
class UNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNNSegmentation.UNet(n_class=config['n_class'], conv_block=config['conv_block'], in_channel_dim=config['in_channel_dim'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=False, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=0.000025, verbose=True)
        return ({
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        })

## RESNET 50
class ResNet50Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNNSegmentation.ResNet(CNNSegmentation.Bottleneck, [3, 4, 6, 3], num_classes = config['n_class'])
        # self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # self.model.classifier = DeepLabHead(2048, 2)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=False, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=0.000025, verbose=True)
        return ({
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        })

class ResNet101Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNNSegmentation.ResNet(CNNSegmentation.Bottleneck, [3, 4, 23, 3], config['n_class'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, nesterov=False, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=0.000025, verbose=True)
        return ({
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        })

# Callback
early_stopping = pl.callbacks.EarlyStopping(
   monitor='val_iou',
   mode='max',
   patience=50,
   verbose=True
)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_iou',
    mode='max',
    verbose=True,
    save_last=True
)

train_transform = A.Compose([
    A.RandomCrop(480,480),
    A.RandomRotate90(),
    A.Flip(),
    A.Resize(256, 256), #HxW tensor dim
])
val_transform = A.Compose([
    A.CenterCrop(480,480),
    A.Resize(256, 256), #HxW tensor dim
])

train_transforms = []
val_transforms = []

for i in range(8):
    if i==0:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=0,std=1)
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=0,std=1)
        ]))
    elif i==1:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0,0),std=(1,1))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0,0),std=(1,1))
        ]))
    elif i==2:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]))
    elif i==3:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0, 0.485, 0.456, 0.406), std=(1, 0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0, 0.485, 0.456, 0.406), std=(1, 0.229, 0.224, 0.225))
        ]))
    elif i==4:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0, 0, 0.485, 0.456, 0.406), std=(1, 1, 0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0, 0, 0.485, 0.456, 0.406), std=(1, 1, 0.229, 0.224, 0.225))
        ]))
    elif i==5:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))
    elif i==6:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(1, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(1, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))
    elif i==7:
        train_transforms.append(A.Compose([
            train_transform,
            A.Normalize(mean=(0, 0, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(1, 1, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))

        val_transforms.append(A.Compose([
            val_transform,
            A.Normalize(mean=(0, 0, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(1, 1, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))
        ]))


# Training
kfold = KFold(5, True, 1)

def get_filelist(path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.splitext(filename)[0])
        return files_list

file_list = get_filelist(os.path.join(dataset_dir, "images"))

if(config['arch'] == "UNet"):
    model = UNetModel()
elif(config['arch'] == "ResNet50"):
    model = ResNet50Model()
elif(config['arch'] == "ResNet101"):
    model = ResNet101Model()

for train, test in kfold.split(file_list):
    data_train = [file_list[i] for i in train]
    data_test = [file_list[i] for i in test]
    train_data = SegmentationDataset(dataset_dir=dataset_dir, data_train=data_train, data_test=data_test, split="train", transform=train_transforms[in_channel-1], in_channel=in_channel)
    val_data = SegmentationDataset(dataset_dir=dataset_dir, data_train=data_train, data_test=data_test, split="valid", transform=val_transforms[in_channel-1], in_channel=in_channel)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4)

    
    trainer = pl.Trainer(gpus=1, max_epochs=epoch, weights_summary='full', callbacks=[checkpoint, early_stopping])
    trainer.fit(model, train_loader, val_loader)
