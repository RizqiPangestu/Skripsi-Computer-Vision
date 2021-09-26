#IMPORT PYTHON PACKAGES
import time
import yaml
import cv2
import torch
import numpy as np
from skimage import feature

#import arsitektur CNN Deep learning
#pelajari di deeplearning/CNNSegmentation.py
from deeplearning import CNNSegmentation

#Color
RED = (0,0,255)
GREEN = (0,255,0)
color = (RED, GREEN)  # RED dan GREEN, sesuaikan jumlah class

#DEFINE DIREKTORI
video_dir = "dataset_unlabelled/papan1.mp4"
model_path = "model/model_jit.pt"
frame_dim = [480, 640] #HxW
input_dim = (256,256)

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

#OPEN CONFIGURATION
with open("config.yml", 'r') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

#load bobot-bobot network
model = torch.jit.load(model_path)
#pindah model ke VRAM GPU atau CPU tergantung cuda availability di atas
model.to(device)
#ganti ke mode eval untuk inference, seperti validation setelah training
model.eval()

print("\n==========================================")
print("OPEN VIDEO CAM......")
cam = cv2.VideoCapture(video_dir)
size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('I','4','2','0'),10, size)
success,_ = cam.read()

print("\n==========================================")
print("INFERENCE ON "+devicename)

while success:
	#ambil waktu mulai
	start = time.time()
	
	#print("frame: ", count)
	# Read frame video
	_,img = cam.read()
	# Load image
	# gx = cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
	# gy = cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
	# gx = cv2.convertScaleAbs(gx)
	# gy = cv2.convertScaleAbs(gy)
	# sobel = cv2.addWeighted(gx,0.5,gy,0.5,0)

	# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# lbp = feature.local_binary_pattern(gray,44,2, method="ror")
	# img = np.dstack((lbp,lbp))
	# img = np.dstack((img,lbp))

	img = cv2.resize(img, (frame_dim[1], frame_dim[0]), interpolation = cv2.INTER_AREA)
	#print(img.shape)

	#augmentasi juga seperti training
	aug_img = cv2.resize(img, (input_dim[1], input_dim[0]), interpolation = cv2.INTER_AREA)
	aug_img = normalize(aug_img)	
	
	#normalisasi dan transpose seperti training
	norm_img = aug_img.astype('float32') / 255
	#transpose array image dan mask menjadi channel first
	# semula default 0H x 1W x 2Channel
	#menjadi channel x H x W, sehingga 2,0,1
	norm_img = norm_img.transpose(2, 0, 1)
	
	#expand dim supaya menjadi batch x channel x h x w
	norm_img = np.expand_dims(norm_img, axis=0)	

	# compute Y_pred
	#masukkan variabel X ke CUDA atau CPU tergantung cuda availability di atas
	X = torch.from_numpy(norm_img).to(device)#cuda()
	#Y_true = torch.from_numpy(mask).to(device)#cuda()
	Y_pred = model(X)
	#pindah tensor Ypred dari GPU ke cpu untuk pemrosesan lebih lanjut
	Y_pred = torch.sigmoid(Y_pred).cpu().detach().numpy()

	print(Y_pred.shape)
	#Y_pred[0][j] berarti mengambil batch ke 0 dan mask class ke j, 2D lainnya adalah H x W
	#dikali 255 berarti dikembalikan ke 8bit dari normalisasi
	#final_mask = []
	img_mask = img.copy()
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
		print_n_obj = config['class_name'][j]+" = "+str(ret-1)
		print(print_n_obj)
		
		res_mask = np.expand_dims(res_mask, axis=-1) #tambahkan channel di axis paling akhir, jadi #H x W x channel seperti image
		if j==0:
			img_mask[:,:,2:3] = img[:,:,2:3] + res_mask #apply mask kelas 0 ke channel RED image
		elif j==1:
    			img_mask[:,:,1:2] = img[:,:,1:2] + res_mask #apply mask kelas 1 ke channel GREEN image
		
		
		#tulis jumlah object di frame
		posisi = [frame_dim[1]-int(frame_dim[1]*0.25), frame_dim[0]-(int(frame_dim[0]*0.06)*(j+1))]
		img_mask = cv2.putText(img_mask, print_n_obj, (posisi[0], posisi[1]), cv2.FONT_HERSHEY_SIMPLEX,  
					1, color[j], 3, cv2.LINE_AA)
	
	#elapsed time
	elapsed = time.time() - start 
	FPS = int(1/elapsed)
	
	#tulis FPS di frame
	printFPS = "FPS: "+str(FPS)
	img_mask = cv2.putText(img_mask, printFPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  
            1, (0, 0, 255), 2, cv2.LINE_AA)
	
	#TAMPILKAN FRAME DAN HASIL RECOGNITION
	print(printFPS)
	cv2.imshow("Welding Inspection with: "+config['arch'], img_mask)
	cv2.imshow("Res mask",res_mask+img)
	cv2.imshow("image",label_hue)
	videoWriter.write(img_mask)
	#cv2.imshow("BINARY FRAME - 1", imagex1)
	#cv2.imshow("BINARY FRAME - 2", imagex2)
	key = cv2.waitKey(1) & 0xFF
 
	# press q to CLOSE WINDOW
	if key == ord("q") or key == ord("Q"):
		break

print("------DONE------")
cv2.destroyAllWindows()
