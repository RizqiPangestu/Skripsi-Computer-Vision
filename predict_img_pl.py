#IMPORT PYTHON PACKAGES
import yaml
import cv2
import torch
import numpy as np

#import arsitektur CNN Deep learning
#pelajari di deeplearning/CNNSegmentation.py
from deeplearning import CNNSegmentation

#Color
RED = (0,0,255)
GREEN = (0,255,0)
color = (RED, GREEN)  # RED dan GREEN, sesuaikan jumlah class

#DEFINE DIREKTORI
model_path = "model/model_jit.pt"
mod_dir = "model/VGG-VGG-VGG-UNet/"
pred_dir = "tools/dataset/predict/"
namefile = "result2021-07-08-135845g1b2.jpg"

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
if config['arch'] == 'NestedUNet':
    model = CNNSegmentation.NestedUNet(n_class=config['n_class'],
            conv_block=config['conv_block'], in_channel_dim=config['tensor_dim'][1]) #ambil channel tensor dim
elif config['arch'] == 'UNet':
    model = CNNSegmentation.UNet(n_class=config['n_class'],
            conv_block=config['conv_block'], in_channel_dim=config['tensor_dim'][1]) #ambil channel tensor dim
elif config['arch'] == 'DeepUNet':
    model = CNNSegmentation.DeepUNet(n_class=config['n_class'],
            conv_block=config['conv_block'], in_channel_dim=config['tensor_dim'][1]) #ambil channel tensor dim

#cek apakah model di train pada multi-GPU (GPU > 1)
#jika iya maka harus diparalel juga ketika load model
if config['n_gpu'] > 1:
    print("MODEL DI TRAIN PADA MULTI-GPU, MAKA MODEL JUGA HARUS DIPARALEL UNTUK INFERENCE")
    model = torch.nn.DataParallel(model)

#load bobot-bobot network
model = torch.jit.load(model_path)
#pindah model ke VRAM GPU atau CPU tergantung cuda availability di atas
model.to(device)
#ganti ke mode eval untuk inference, seperti validation setelah training
model.eval()

print("\n==========================================")
print("INFERENCE ON "+devicename)
# BACA IMAGE 
img = cv2.imread(pred_dir+namefile)
gx = cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
gy = cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
gx = cv2.convertScaleAbs(gx)
gy = cv2.convertScaleAbs(gy)
sobel = cv2.addWeighted(gx,0.5,gy,0.5,0)
img = sobel
print(img.shape) #H X W X Channel
#img = cv2.resize(img, (config['input_w'], config['input_h'])) #W x H

#buat frame mask
masker = np.zeros((config['n_class'],3,img.shape[0], img.shape[1]) ,dtype=int) #classxChxHxW
#ubah warnanya sesuai colormap class
for i in range(config['n_class']):
    for j in range(masker.shape[1]):
        masker[i][j] = masker[i][j] + color[i][j]

"""
#DAN BUAT FRAME MASKNYA
mask = []
for i in range(config['num_classes']): #append mask untuk setiap class yg ada
    mask.append(cv2.imread(pred_dir+"mask.jpg",cv2.IMREAD_GRAYSCALE)[..., None])
mask = np.dstack(mask)
"""

#DITUTUP KARENA DIJETSON NANO GA BISA
#JADI AUGMENTASI MANUAL TIDAK PAKAI LIBRARY
"""
#augmentasi juga seperti training
transforming = Compose([
    transforms.Resize(config['tensor_dim'][2], config['tensor_dim'][3]), #HxW tensor dim
    transforms.Normalize()])
augmented = transforming(image=img)#, mask=mask)
aug_img = augmented['image']
#mask = augmented['mask']
"""

#augmentasi juga seperti training
#KARENA DIJETSON NANO GA BISA
#JADI AUGMENTASI MANUAL TIDAK PAKAI LIBRARY ALBUMENTATIONS
aug_img = cv2.resize(img, (config['tensor_dim'][3], config['tensor_dim'][2]), interpolation = cv2.INTER_AREA)
aug_img = normalize(aug_img)

#print(aug_img.shape)
#print(np.max(aug_img))
#print(np.min(aug_img))
#cv2.imwrite("x.jpg",aug_img) #cetak predicted mask

#normalisasi dan transpose seperti training
norm_img = aug_img.astype('float32') / 255
#transpose array image dan mask menjadi channel first
# semula default 0H x 1W x 2Channel
#menjadi channel x H x W, sehingga 2,0,1
norm_img = norm_img.transpose(2, 0, 1)
#mask = mask.astype('float32') / 255
#mask = mask.transpose(2, 0, 1)
    
#expand dim supaya menjadi batch x channel x h x w
norm_img = np.expand_dims(norm_img, axis=0)	
print(norm_img.shape)
#mask = np.expand_dims(mask, axis=0)
#print(mask.shape)

# compute Y_pred
#masukkan variabel X ke CUDA atau CPU tergantung cuda availability di atas
X = torch.from_numpy(norm_img).to(device)#cuda()
#Y_true = torch.from_numpy(mask).to(device)#cuda()
Y_pred = model(X)
#pindah tensor Ypred dari GPU ke cpu untuk pemrosesan lebih lanjut
Y_pred = torch.sigmoid(Y_pred).cpu().detach().numpy()
    
#iou = iou_score(Y_pred, Y_true)
#avg_meter = AverageMeter()
#avg_meter.update(iou, input.size(0))
#print('IoU: %.4f' % avg_meter.avg)

#print(Y_pred.shape)
#Y_pred[0][j] berarti mengambil batch ke 0 dan mask class ke j, 2D lainnya adalah H x W
#dikali 255 berarti dikembalikan ke 8bit dari normalisasi
#final_mask = []

#all_mask = []
imgx = img

for j in range(config['n_class']):
    pred_mask = (Y_pred[0][j] * 255).astype('uint8')
    res_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0])) #WxH
    
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
    print(print_n_obj)
    cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_inst_mask"+str(j)+".jpg",labeled_img) #cetak predicted mask
    
    #CETAK MASK YANG SAMA UNTUK 1 CLASS
    #operasi and, yang 1 dipertahankan, yang lain di 0 kan
    color_mask = masker[j] * res_mask
    color_mask = color_mask.transpose(1,2,0) #ganti channel last, HxWxCh
    
    cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_sem_mask"+str(j)+".jpg",color_mask) #cetak predicted mask
    
    #all_mask.append(res_mask)
    
    #res_mask = np.expand_dims(res_mask, axis=-1) #tambahkan channel di axis paling akhir, jadi #H x W x channel seperti image
    #print(res_mask.shape)
    imgx = imgx+color_mask
    img_mask = img + color_mask #apply mask ke image
    print(img_mask.shape)
    #out = cv2.bitwise_not(img, img, mask = res_mask)
    cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_img"+str(j)+".jpg",img_mask)
    
    #tulis jumlah object di frame
    posisi = [img.shape[1]-int(img.shape[1]*0.175), img.shape[0]-(int(img.shape[0]*0.05)*(j+1))]
    imgx = cv2.putText(np.float32(imgx), print_n_obj, (posisi[0], posisi[1]), cv2.FONT_HERSHEY_SIMPLEX,  
            4, (color[j][0], color[j][1], color[j][2]), 10, cv2.LINE_AA)
    
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
"""
with open('toyotav2-mask.npy', 'wb') as f:
    np.save(f, all_mask)
"""

cv2.imwrite(pred_dir+config["conv_block"][0]+"-"+config["conv_block"][1]+"-"+config["conv_block"][2]+"-"+config["arch"]+"_"+namefile+"_img.jpg",imgx)
