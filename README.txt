0. HOW TO USE THIS TEMPLATE
	- jika memakai local machine,
		- pastikan punya GPU NVIDIA yang ada CUDA corenya
			- tidak yakin? -> search di web NVIDIA sendiri
		- ikuti langkah mulai dari 1 - 6
	- jika memakai cloud (AWS, GCP, atau cloud service lain),
		- masing-masing cloud service punya caranya sendiri, pelajari di webnya masing-masing
		- buat VM instance sesuai kebutuhan, sesuaikan environmentnya juga
			- saran: pakai pre-config environment yang telah disediakan
		- upload semua file project ke VM instance
		- langsung masuk ke langkah 4 - 6

1. INSTALL PYTHON
	- kunjungi: https://www.python.org/downloads/
	- install python versi 3.7.x saja

2. INSTALL NVIDIA DRIVER, CUDA TOOLKIT, DAN CUDNN YANG SESUAI
	- NVIDIA Driver: sesuaikan dengan seri GPU dan OS
		- https://www.nvidia.com/Download/index.aspx
	- CUDA Toolkit dan cuDNN
		- cek compability dengan driver dulu:
			- https://docs.nvidia.com/deploy/cuda-compatibility/index.html
		- lalu pilih CUDA Toolkit dan CuDNN yang sesuai (10.0 atau 10.1 saja kalau bisa)
			- windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
			- linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

3. INSTALL PYTORCH GPU dan TORCHVISION
	- kunjungi: https://download.pytorch.org/whl/torch_stable.html
	- cari package yang sesuai dengan environment di komputer
		- perhatikan versi cuda toolkit
		- versi python
		- OS yang dipakai
		- pilih versi pytorch
		- lalu copy linknya
	- pip install <link file .whl>
	
4. INSTALL SEMUA PACKAGE YANG DIPERLUKAN
	- aktifkan cmd atau buka terminal di direktori project
	- connect ke internet
	- pip install -r requirements.txt
	- (perhatikan link whl 2 baris terakhir, perlu diinstall lagi atau tidak)
	
5. TRAINING
	- aktifkan cmd atau buka terminal di direktori project
	- prepare dataset (langkah ini bisa di SKIP, karena sudah dilakukan, 1x saja)
		- python csv2binmask.py
			- program ini mengubah annotasi polygon di CSV pada folder annotation
			- dan image-image .jpg di folder images
			- menjadi file mask (hitam-putih) dalam bentuk .jpg di folder masks
			- perhatikan file frame yang di skip pada csv2binmask.py, sesuaikan kebutuhan
		- untuk pembuatan file .csv berisi polygon menggunakan VGG Image Annotator (VIA) Tools
			- http://www.robots.ox.ac.uk/~vgg/software/via/
	- baca dulu config di train.py, setting konfigurasi sesuai yang diinginkan
	- python train.py
	- training dimulai, dan tergenerate file2 sbg berikut di model/namamodel:
		- model_config.yml -> settingan konfigurasi training
		- model_summary.txt -> summary shape tiap layer pada model
		- model_graph.png -> visualisasi model
		- model_graph -> text file berisi konfigurasi visualisasi model
		- model_log -> history training model, berisi:
			- epoch, lrate, train_loss, val_loss
			- train_dice,val_dice,train_iou,val_iou,elapsed_time
		- model_weights.pth -> file bobot model yang sudah disave
			- bisa juga mensave model tiap epoch, baca setting konfigurasi
		- data_info.yml -> file info dataset yang digunakan

6. INFERENCE
	- aktifkan cmd atau buka terminal di direktori project
	- untuk prediksi image
		- python predict_img.py (perhatikan model dan gambar yang diload)
		- file prediksi akan tersimpan di prediction/image
	- untuk prediksi video
		- python predict_vid.py (perhatikan model dan gambar yang diload)
		- file prediksi akan tersimpan di prediction/video

7. SETUP JETSON AGX XAVIER
	- siapkan OS Ubuntu 18.04 pada host
	- install Nvidia SDK Manager pada host
		- https://developer.nvidia.com/embedded/jetpack)
	- ikuti langkah yang ada pada https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html
		- jika hanya ingin melakukan flashing jetson tidak perlu mencentang host machine
		- selama instalasi jetson harus terhubung dengan keyboard, mouse, dan monitor
		- gunakan user dan password berikut
			- username: toyota
			- password: ugmtoyota
	- restart jetson
	- install pytorch
		- https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048)
	- install pyyaml
		- pip3 install pyyaml
	- install imutils
		- pip3 install imutils
	- copy file program ke jetson (bisa menggunakan sdcard)

	- untuk mempermudah development headless tanpa monitor (opsional)
		- setup koneksi wifi agar bisa konek otomatis ke SSID yang diinginkan
		- setelah login aktifkan automatic login pada System Settings -> User Accounts -> Automatic Login
		- aktifkan vnc server dengan mengikuti instruksi pada folder L4T-README di desktop
		- gunakan SSH atau VNC untuk mengakses jetson
			- bisa menggunakan bonjour/zeroconf agar tidak perlu menggunakan ip static/mencari ip jetson secara manual

8. INFERENCE PADA JETSON AGX XAVIER
	- buka terminal di direktori project
	- untuk prediksi image
		- python3 predict_img.py (perhatikan model dan gambar yang diload)
		- file prediksi akan tersimpan di prediction/image
	- untuk prediksi video
		- python3 predict_vid.py (perhatikan model dan gambar yang diload)
		- file prediksi akan tersimpan di prediction/video
	- untuk prediksi menggunakan webcam
		- python3 predict_real.py (perhatikan model dan gambar yang diload)
		- akan muncul window live view