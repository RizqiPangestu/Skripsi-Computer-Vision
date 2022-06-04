#IMPORT PYTHON PACKAGES
import yaml
import cv2
import torch
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl

#import arsitektur CNN Deep learning
#pelajari di deeplearning/CNNSegmentation.py
from deeplearning import CNNSegmentation

#Color
RED = (0,0,255)
GREEN = (0,255,0)
color = (RED, GREEN)  # RED dan GREEN, sesuaikan jumlah class

#DEFINE DIREKTORI
model_path = "model/model_jit.pt"
mod_dir = "model/"
pred_dir = "dataset/images/"
namefile = "2021-07-08-135743g1b1.jpg"

dataset_path = "dataset_old"
images = []
goods = []
bads = []

for filename in os.listdir(dataset_path + "/images"):
    images.append(filename)

#cek apakah akan menggunakan GPU untuk inference atau tidak
if torch.cuda.is_available():
#set device GPU
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    devicename = "GPU"
else:
    device = torch.device("cpu")
    devicename = "CPU"

#AMBIL DARI ALBUMENTATIONS
#https://albumentations.readthedocs.io/en/latest/_modules/albumentations/augmentations/transforms.html#Normalize
#KARENA DIJETSON NANO GA BISA
#JADI AUGMENTASI MANUAL TIDAK PAKAI LIBRARY
def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

#PRINT CONFIGURATION
print("==========================================")
print("MODEL CONFIGURATION:")
with open(mod_dir+"model_config.yml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
for key in config.keys():
    print('%s: %s' % (key, str(config[key])))
#baca data info datanya
with open(mod_dir+"data_info.yml", 'r') as f:
    data_info = yaml.load(f, Loader=yaml.FullLoader)

# LOAD ARSITEKTUR DAN WEIGHTS MODEL
print("\n==========================================")
print("IMPORT ARSITEKTUR DL: "+config['arch']+" DENGAN CONV BLOCK: "+str(config['conv_block'])+" DAN COMPILE")
#buat arsitektur
if config['arch'] == 'UNet':
    ckpt_path = "lightning_logs/UNet/checkpoints/epoch=260-step=12788.ckpt"
elif config['arch'] == 'ResNet50':
    ckpt_path = "lightning_logs/ResNet50/checkpoints/epoch=321-step=8049.ckpt"
elif config['arch'] == 'ResNet101':
    ckpt_path = "lightning_logs/ResNet101/checkpoints/epoch=225-step=5649.ckpt"

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #buat arsitektur
        if config['arch'] == 'UNet':
            self.model = CNNSegmentation.UNet(n_class=config['n_class'], conv_block=config['conv_block'], in_channel_dim=config['in_channel_dim']) #ambil channel tensor dim
        elif config['arch'] == 'ResNet50':
            self.model = CNNSegmentation.ResNet(CNNSegmentation.Bottleneck, [3, 4, 6, 3], num_classes = config['n_class'])
        elif config['arch'] == 'ResNet101':
            self.model = CNNSegmentation.ResNet(CNNSegmentation.Bottleneck, [3, 4, 23, 3], num_classes=config['n_class'])
    
    def forward(self, x):
        return self.model(x)

#load bobot-bobot network
# model = torch.jit.load(model_path)
model = SegmentationModel()
model = model.load_from_checkpoint(ckpt_path)
#cek apakah model di train pada multi-GPU (GPU > 1)
#jika iya maka harus diparalel juga ketika load model
if config['n_gpu'] > 1:
    print("MODEL DI TRAIN PADA MULTI-GPU, MAKA MODEL JUGA HARUS DIPARALEL UNTUK INFERENCE")
    model = torch.nn.DataParallel(model)
#pindah model ke VRAM GPU atau CPU tergantung cuda availability di atas
model.to(device)
#ganti ke mode eval untuk inference, seperti validation setelah training
model.eval()

print("\n==========================================")
print("INFERENCE ON "+devicename)
# BACA IMAGE 
image_container = []
mask_container = []
for filename in images:
    image = cv2.imread(os.path.join(dataset_path,"images",filename))
    image_container.append(image)
    # Load Label
    mask_bad = cv2.imread(os.path.join(dataset_path, "masks", "bad", filename.split(".")[0] + ".png"),cv2.IMREAD_GRAYSCALE)
    if(mask_bad is not None):
        bads.append(mask_bad)
    else:
        bads.append(np.zeros((image.shape[0],image.shape[1])))

    mask_good = cv2.imread(os.path.join(dataset_path, "masks", "good", filename.split(".")[0] + ".png"),cv2.IMREAD_GRAYSCALE)
    if(mask_good is not None):
        goods.append(mask_good)
    else:
        goods.append(np.zeros((image.shape[0],image.shape[1])))
mask_container.append(bads)
mask_container.append(goods)

print(image_container[0].shape) #H X W X Channel
print(mask_container[0][0].shape)
print(mask_container[0][0].shape)
#img = cv2.resize(img, (config['input_w'], config['input_h'])) #W x H

masker_container = []
for index in range(len(image_container)):
    #buat frame mask
    masker = np.zeros((config['n_class'],3,image_container[index].shape[0], image_container[index].shape[1]) ,dtype=int) #classxChxHxW
    #ubah warnanya sesuai colormap class
    for i in range(config['n_class']):
        for j in range(masker.shape[1]):
            masker[i][j] = masker[i][j] + color[i][j]
    
    masker_container.append(masker)

def center_crop(img,crop):
    center = (img.shape[1]//2,img.shape[0]//2)
    x_new = center[1] - (crop[1]//2)
    y_new = center[0] - (crop[0]//2)

    return img[y_new:y_new+crop[1],x_new:x_new+crop[0]]

#augmentasi juga seperti training
def aug_image(img):
    aug_img = center_crop(img,(480,480))
    aug_img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    aug_img = normalize(aug_img)
    return aug_img

aug_img_container = []
for i in range(len(image_container)):
    aug_img_container.append(aug_image(image_container[i]))

#print(aug_img.shape)
#print(np.max(aug_img))
#print(np.min(aug_img))
#cv2.imwrite("x.jpg",aug_img) #cetak predicted mask

#normalisasi dan transpose seperti training
norm_img_container = []
for i in range(len(image_container)):
    norm_img = aug_img_container[i].astype('float32') / 255
    #transpose array image dan mask menjadi channel first
    # semula default 0H x 1W x 2Channel
    #menjadi channel x H x W, sehingga 2,0,1
    norm_img = norm_img.transpose(2, 0, 1)
    #mask = mask.astype('float32') / 255
    #mask = mask.transpose(2, 0, 1)
        
    #expand dim supaya menjadi batch x channel x h x w
    norm_img = np.expand_dims(norm_img, axis=0)	
    norm_img_container.append(norm_img)
print("Norm Image Shape",norm_img_container[0].shape)
#mask = np.expand_dims(mask, axis=0)
#print(mask.shape)

# compute Y_pred
Y_pred_container = []
for i in range(len(image_container)):
    #masukkan variabel X ke CUDA atau CPU tergantung cuda availability di atas
    X = torch.from_numpy(norm_img_container[i]).to(device)#cuda()
    #Y_true = torch.from_numpy(mask).to(device)#cuda()
    Y_pred = model(X)
    #pindah tensor Ypred dari GPU ke cpu untuk pemrosesan lebih lanjut
    Y_pred = torch.sigmoid(Y_pred).cpu().detach().numpy()
    Y_pred_container.append(Y_pred)
    
#iou = iou_score(Y_pred, Y_true)
#avg_meter = AverageMeter()
#avg_meter.update(iou, input.size(0))
#print('IoU: %.4f' % avg_meter.avg)

print("Pred Shape = ",len(Y_pred_container), Y_pred_container[0].shape)
#Y_pred[0][j] berarti mengambil batch ke 0 dan mask class ke j, 2D lainnya adalah H x W
#dikali 255 berarti dikembalikan ke 8bit dari normalisasi
#final_mask = []


# cf_matrix = confusion_matrix(y_true, y_pred)

#all_mask = []
imgx_container = []
#good
accuracy_container_good = []
recall_container_good = []
precision_container_good = []
f1score_container_good = []
confussion_matrix_good = []
#bad
accuracy_container_bad = []
recall_container_bad = []
precision_container_bad = []
f1score_container_bad = []
confussion_matrix_bad = []

#revision confusion matrix
pred_tp_bad = 0
pred_tn_bad = 0
pred_fp_bad = 0
pred_fn_bad = 0

pred_tp_good = 0
pred_tn_good = 0
pred_fp_good = 0
pred_fn_good = 0

for i in range(len(image_container)):
    imgx_container.append(image_container[i])

def normalize_img(img):
    img = np.where(img>127, 1, 0)
    return img

for i in range(len(image_container)):
    img_temp = []
    n_obj_container = []
    for j in range(config['n_class']):
        pred_mask = (Y_pred_container[i][0][j] * 255).astype('uint8')
        res_mask = cv2.resize(pred_mask, (image_container[i].shape[1], image_container[i].shape[0])) #WxH
        
        #CETAK MASK UNTUK SETIAP OBJECT UNTUK PERHITUNGAN JUMLAH OBJECT
        #deteksi berapa object dalam 1 frame
        # frame class 0 sendiri, frame class 1 sendiri
        ret, labels = cv2.connectedComponents(res_mask)
        label_hue = np.uint8(179 * labels / np.max(labels))
        #buat blank channel untuk di gabung
        blank_ch = 255 * np.ones_like(label_hue)
        #gabung ketiga channel sehingga jadi WxHxchannel
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        #dikonversi ke RGB untuk ditampilkan
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        #set background jadi 0 semua, hitam
        labeled_img[label_hue == 0] = 0
        
        #cetak
        print_n_obj = data_info['class_name'][j]+" = "+str(ret-1)
        n_obj_container.append(ret-1)
        # cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_inst_mask"+str(j)+".jpg",labeled_img) #cetak predicted mask
        
        #CETAK MASK YANG SAMA UNTUK 1 CLASS
        #operasi and, yang 1 dipertahankan, yang lain di 0 kan
        color_mask = masker[j] * res_mask
        color_mask = color_mask.transpose(1,2,0) #ganti channel last, HxWxCh
        # print("color_mask", color_mask.shape)
        
        # cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_sem_mask"+str(j)+".jpg",color_mask) #cetak predicted mask
        
        #all_mask.append(res_mask)
        
        #res_mask = np.expand_dims(res_mask, axis=-1) #tambahkan channel di axis paling akhir, jadi #H x W x channel seperti image
        #print(res_mask.shape)
        imgx_container[i] = imgx_container[i]+color_mask
        img_mask = image_container[i] + color_mask #apply mask ke image
        # print("img mask",img_mask.shape)
        #out = cv2.bitwise_not(img, img, mask = res_mask)
        # cv2.imshow(f"img_mask{j}",np.array(img_mask, dtype=np.uint8))
        # cv2.imshow(f"color_mask{j}",np.array(color_mask, dtype=np.uint8))
        # cv2.imshow(f"res_mask{j}",res_mask)
        # cv2.imshow(f"mask",mask_container[j][i])

        img_temp.append(np.array(img_mask, dtype=np.uint8))
        thresh = 127
        res_mask = cv2.threshold(res_mask, 127, 255, cv2.THRESH_BINARY)[1]
        res_mask = normalize_img(res_mask)
        ret, labels = cv2.connectedComponents(np.uint8(res_mask))
        mask_container[j][i] = normalize_img(mask_container[j][i])
        smoothies = 0.000000000001
        FP = len(np.where(res_mask - mask_container[j][i]  == 1)[0])
        FN = len(np.where(res_mask - mask_container[j][i]  == -1)[0])
        TP = len(np.where(res_mask + mask_container[j][i] == 2)[0])
        TN = len(np.where(res_mask + mask_container[j][i] == 0)[0])

        IoU = TP/(FP+FN+TP+smoothies)
        
        # print(data_info['class_name'][j],FP,FN,TP,TN,FP+FN+TP+TN)
        accuracy = (TP+TN)/(TP+TN+FP+FN+smoothies)
        recall = TP/(TP+FN+smoothies)
        precision = TP/(TP+FP+smoothies)
        if(j == 0):
            confussion_matrix_bad.append([[TP,FP],[FN,TN]])
            accuracy_container_bad.append(accuracy)
            recall_container_bad.append(recall)
            precision_container_bad.append(precision)
            f1score_container_bad.append(2*(recall*precision)/(recall+precision+smoothies))

            if(IoU >= 0.5 and ret-1 > 0):
                pred_tp_bad += 1
            elif(IoU >= 0.5 and ret-1 == 0):
                pred_fp_bad += 1
            elif(IoU < 0.5 and ret-1 > 0):
                pred_fn_bad += 1
            elif(IoU < 0.5 and ret-1 == 0):
                pred_tn_bad += 1
        else:
            confussion_matrix_good.append([[TP,FP],[FN,TN]])
            accuracy_container_good.append(accuracy)
            recall_container_good.append(recall)
            precision_container_good.append(precision)
            f1score_container_good.append(2*(recall*precision)/(recall+precision+smoothies))

            if(IoU >= 0.5 and ret-1 > 0):
                pred_tp_good += 1
            elif(IoU >= 0.5 and ret-1 == 0):
                pred_fp_good += 1
            elif(IoU < 0.5 and ret-1 > 0):
                pred_fn_good += 1
            elif(IoU < 0.5 and ret-1 == 0):
                pred_tn_good += 1
        
        # cv2.waitKey(0) 
  
        # #closing all open windows 
        # cv2.destroyAllWindows()
        
        #tulis jumlah object di frame
        # posisi = [image_container[i].shape[1]-int(image_container[i].shape[1]*0.175), image_container[i].shape[0]-(int(image_container[i].shape[0]*0.05)*(j+1))]
        # imgx = cv2.putText(np.float32(imgx), print_n_obj, (posisi[0], posisi[1]), cv2.FONT_HERSHEY_SIMPLEX,  
        #         4, (color[j][0], color[j][1], color[j][2]), 10, cv2.LINE_AA)
        
        #output = ((0.4 * img) + (0.6 * res_mask*255)).astype("uint8")
        #cv2.imwrite("wkwkwk"+str(j)+".jpg",output)
        #print(output)
        """
        #append semua mask
        final_mask.append(np.array(res_mask))
        final_mask = np.array(final_mask)
        print(final_mask.shape)
        #res_mask = res_mask*255
        
        #apply mask ke image
        for k in range(img.shape[2]): #loop sebanyak channel
            img_mask = img + final_mask.transpose(1,2,0) #transpose jadi channel last sma kyk image
        cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_img.jpg",img_mask)
        """
    # if(n_obj_container[0] > n_obj_container[1]):
    #     cv2.imshow(f"bad",img_temp[0])
    #     print(n_obj_container)

    #     cv2.waitKey(0) 
  
    #     #closing all open windows 
    #     cv2.destroyAllWindows()
    # else:
    #     cv2.imshow(f"good",img_temp[1])
    #     print(n_obj_container)

    #     cv2.waitKey(0) 
  
    #     #closing all open windows 
    #     cv2.destroyAllWindows()
"""
with open('toyotav2-mask.npy', 'wb') as f:
    np.save(f, all_mask)
""" 
# for i in n_obj_container:
#     print(i)
# cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_img.jpg",imgx)

accuracy = sum(accuracy_container_bad)/len(accuracy_container_bad)
recall = sum(recall_container_bad)/len(recall_container_bad)
precision = sum(precision_container_bad)/len(precision_container_bad)
f1score = sum(f1score_container_bad)/len(f1score_container_bad)
print("Total image test for Bad Label =", len(accuracy_container_bad))
print("Accuracy =",accuracy)
print("Recall =",recall)
print("Precision =",precision)
print("F1 Score =",f1score)

accuracy = sum(accuracy_container_good)/len(accuracy_container_good)
recall = sum(recall_container_good)/len(recall_container_good)
precision = sum(precision_container_good)/len(precision_container_good)
f1score = sum(f1score_container_good)/len(f1score_container_good)
print("Total image test for Good Label=", len(accuracy_container_good))
print("Accuracy =",accuracy)
print("Recall =",recall)
print("Precision =",precision)
print("F1 Score =",f1score)

cf_tp = 0
cf_fp = 0
cf_tn = 0
cf_fn = 0
for i in range(len(confussion_matrix_good)):
    cf_tp += confussion_matrix_good[i][0][0]
    cf_fp += confussion_matrix_good[i][0][1]
    cf_fn += confussion_matrix_good[i][1][0]
    cf_tn += confussion_matrix_good[i][1][1]

cf_tp /= len(confussion_matrix_good)
cf_fp /= len(confussion_matrix_good)
cf_tn /= len(confussion_matrix_good)
cf_fn /= len(confussion_matrix_good)

# cf_matrix = [[round(cf_tp),round(cf_fn)],[round(cf_fp),round(cf_tn)]]
# print(cf_matrix)

# ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

# ax.set_title('Confusion Matrix for Good Labels\n\n')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['True','False'])
# ax.yaxis.set_ticklabels(['True','False'])

# ## Display the visualization of the Confusion Matrix.
# plt.show()

for i in range(len(confussion_matrix_bad)):
    cf_tp += confussion_matrix_bad[i][0][0]
    cf_fp += confussion_matrix_bad[i][0][1]
    cf_fn += confussion_matrix_bad[i][1][0]
    cf_tn += confussion_matrix_bad[i][1][1]

cf_tp /= len(confussion_matrix_bad)
cf_fp /= len(confussion_matrix_bad)
cf_tn /= len(confussion_matrix_bad)
cf_fn /= len(confussion_matrix_bad)

# cf_matrix = [[round(cf_tp),round(cf_fn)],[round(cf_fp),round(cf_tn)]]
# print(cf_matrix)

# ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

# ax.set_title('Confusion Matrix for Bad Labels')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')

# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['True','False'])
# ax.yaxis.set_ticklabels(['True','False'])

# ## Display the visualization of the Confusion Matrix.
# plt.show()

pred_accuracy_bad = (pred_tp_bad+pred_tn_bad)/(pred_tp_bad+pred_tn_bad+pred_fn_bad+pred_fp_bad)
pred_recall_bad = pred_tp_bad/(pred_tp_bad+pred_fn_bad)
pred_precision_bad = pred_tp_bad/(pred_tp_bad+pred_fp_bad)
pred_f1_bad = 2 * pred_recall_bad*pred_precision_bad / (pred_recall_bad+pred_precision_bad)

cf_matrix = [[round(pred_tp_bad),round(pred_fn_bad)],[round(pred_fp_bad),round(pred_tn_bad)]]
print(cf_matrix)

ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

ax.set_title('Confusion Matrix for Bad Labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

## Display the visualization of the Confusion Matrix.
plt.show()

pred_accuracy_good = (pred_tp_good+pred_tn_good)/(pred_tp_good+pred_tn_good+pred_fn_good+pred_fp_good)
pred_recall_good = pred_tp_good/(pred_tp_good+pred_fn_good)
pred_precision_good = pred_tp_good/(pred_tp_good+pred_fp_good)
pred_f1_good = 2 * pred_recall_good*pred_precision_good / (pred_recall_good+pred_precision_good)

cf_matrix = [[round(pred_tp_good),round(pred_fn_good)],[round(pred_fp_good),round(pred_tn_good)]]
print(cf_matrix)

ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

ax.set_title('Confusion Matrix for Good Labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

## Display the visualization of the Confusion Matrix.
plt.show()

print("Total image test for Bad Label =", len(image_container))
print("Accuracy =",pred_accuracy_bad)
print("Recall =",pred_recall_bad)
print("Precision =",pred_precision_bad)
print("F1 Score =",pred_f1_bad)

print("Total image test for Good Label=", len(image_container))
print("Accuracy =",pred_accuracy_good)
print("Recall =",pred_recall_good)
print("Precision =",pred_precision_good)
print("F1 Score =",pred_f1_good)