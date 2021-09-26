import cv2
import numpy as np
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml

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
    def __init__(self, dataset_dir, split="train", val_size=None, seed=None, transform=None,in_channel=None):
        self.dataset_dir = dataset_dir
        self.lbp_dir = 'dataset_lbp'
        self.sobel_dir = 'dataset_sobel'
        self.gray_dir = 'dataset_gray'
        self.transform = transform
        self.in_channel = in_channel
        self.file_list = self.get_filelist(os.path.join(self.dataset_dir, "images"))
        train_list, valid_list = train_test_split(self.file_list, shuffle=True, test_size=val_size, random_state=seed)
        if split == "train":
            self.file_list = train_list
        elif split == "valid":
            self.file_list = valid_list
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        # Load image
        # SELAIN 3 DAN 1 CHANNEL MERUPAKAN KOMBINASI
        if self.in_channel == 1:
            image = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image,axis=2)
        elif self.in_channel == 2:
            image1 = cv2.imread(os.path.join(self.lbp_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(self.gray_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image = np.dstack((image1,image2))
        elif self.in_channel == 3:
            image = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
        elif self.in_channel == 4:
            image1 = cv2.imread(os.path.join(self.lbp_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
            image = np.dstack((image1,image2))
        elif self.in_channel == 5:
            image1 = cv2.imread(os.path.join(self.lbp_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(self.gray_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image3 = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
            image = np.dstack((image1,image2,image3))
        elif self.in_channel == 6:
            image1 = cv2.imread(os.path.join(self.sobel_dir, "images", filename + ".jpg"))
            image2 = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
            image = np.dstack((image1,image2))
        elif self.in_channel == 7:
            image1 = cv2.imread(os.path.join(self.lbp_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(self.sobel_dir, "images", filename + ".jpg"))
            image3 = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
            image = np.dstack((image1,image2,image3))
        elif self.in_channel == 8:
            image1 = cv2.imread(os.path.join(self.lbp_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(self.gray_dir, "images", filename + ".jpg"),cv2.IMREAD_GRAYSCALE)
            image3 = cv2.imread(os.path.join(self.sobel_dir, "images", filename + ".jpg"))
            image4 = cv2.imread(os.path.join(self.dataset_dir, "images", filename + ".jpg"))
            image = np.dstack((image1,image2,image3,image4))
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

    def get_filelist(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.splitext(filename)[0])
        return files_list

# Loss Function
def BCEDiceLoss(Yp, Yt):
    #hitung BCE with logits loss dulu
    bce = torch.nn.functional.binary_cross_entropy_with_logits(Yp, Yt.type_as(Yp))
    #perbedaan BCE biasa dengan BCE with logits
    # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/5
    #BCE biasa belum disigmoid, sementara BCE with logits sudah disigmoid diinternal functionnnya
    #sehingga yang digunakan adalah BCE with logits karena akan di mix dengan Dice yang juga perlu sigmoid
    smooth = 1e-5 #toleransi smooth
    #hitung dice loss, refer: https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
    Yp = torch.sigmoid(Yp)
    num = Yt.size(0)
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = Yp.view(num, -1)
    Yt = Yt.view(num, -1)
    intersection = (Yp * Yt)
    dice = (2. * intersection.sum(1) + smooth) / (Yp.sum(1) + Yt.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    #kalkulasi lossnya
    bce_dice_loss = 0.5 * bce + dice
    return bce_dice_loss

# Metric
#1. PERHITUNGAN IoU
#refer: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou_score(Yp, Yt):
    smooth = 1e-5 #toleransi smooth
    #data sebelumnya telah dijadikan tensor untuk dibaca di GPU
    #dikembalikan ke CPU (numpy) dulu untuk perhitungan IoU
    Yp = torch.sigmoid(Yp).data.cpu().numpy()
    Yt = Yt.data.cpu().numpy()
    #threshold confidence 0.5 aja yang akan diproses
    output_ = Yp > 0.5
    target_ = Yt > 0.5
    intersection = (output_ & target_).sum() #irisan
    union = (output_ | target_).sum() #union
    #rumus IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou
    
#2. PERHITUNGAN DICE COEFFICIENT
#refer: https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_coef(Yp, Yt):
    smooth = 1e-5 #toleransi smooth
    #data sebelumnya telah dijadikan tensor untuk dibaca di GPU
    #maka dikembalikan ke CPU (numpy) dulu untuk perhitungan IoU
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = torch.sigmoid(Yp).view(-1).data.cpu().numpy()
    Yt = Yt.view(-1).data.cpu().numpy()
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice = (2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth)
    return dice

# Model
class SegmentationModel(pl.LightningModule):
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
        self.log('train_loss', loss)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_dice', dice, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = BCEDiceLoss(out, y)
        iou = iou_score(out, y)
        dice = dice_coef(out, y)
        self.log('val_loss', loss)
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

# # Augmentation
# train_transform_4ch = A.Compose([
#     train_transform,
#     A.Normalize(mean=(0.485, 0.456, 0.406, 0), std=(0.229, 0.224, 0.225))
# ])
# val_transform_4ch = A.Compose([
#     val_transform,
#     A.Normalize(mean=(0.485, 0.456, 0.406 ,0), std=(0.229, 0.224, 0.225)),
# ])

# train_transform_3ch = A.Compose([
#     train_transform,
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])
# val_transform_3ch = A.Compose([
#     val_transform,
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])

# train_transform_1ch = A.Compose([
#     train_transform,
#     A.Normalize(mean=0,std=1)
# ])
# val_transform_1ch = A.Compose([
#     val_transform,
#     A.Normalize(mean=0,std=1)
# ])

# if config['in_channel_dim'] == 1:
#     train_transform = train_transform_1ch
#     val_transform = val_transform_1ch
# elif config['in_channel_dim'] == 3:
#     train_transform = train_transform_3ch
#     val_transform = val_transform_3ch



# Training
train_data = SegmentationDataset(dataset_dir=dataset_dir, split="train", val_size=val_size, seed=random_seed, transform=train_transforms[in_channel-1], in_channel=in_channel)
val_data = SegmentationDataset(dataset_dir=dataset_dir, split="valid", val_size=val_size, seed=random_seed, transform=val_transforms[in_channel-1], in_channel=in_channel)

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4)

model = SegmentationModel()
trainer = pl.Trainer(gpus=0, max_epochs=epoch, weights_summary='full', callbacks=[checkpoint, early_stopping])
trainer.fit(model, train_loader, val_loader)
