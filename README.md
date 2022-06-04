# welding-inspection-cv
Repository Computer Vision Welding Inspection System 

## Baca `README.txt` untuk file readme versi lama, readme versi baru masih On Progress

## Hardware Requirment:
- Komputer dengan GPU Nvidia
- Nvidia Jetson AGX (Optional)
- Webcam/Kamera (Optional)

## Software Requirment:
Idealnya software di development PC sama dengan di Jetson AGX dikarenakan tidak semua software support di Jetson AGX (Jetson menggunakan ARM bukan x86), gunakan Conda atau Docker untuk mempermudah proses instalasi.

Untuk versi software yang digunakan disusun berdasarkan Jetpack 4.5.1 (https://developer.nvidia.com/embedded/jetpack)

### Software yang terdapat di Jetson:
- Ubuntu 18.04 (Seharusnya bisa di Ubuntu 20.04)
- Python 3.6
- CUDA 10.2
- cuDNN 8.0
- OpenCV 4.1.1
- TensorRT 7.1.3 (Optional, disarankan menggunakan docker)

### Library Python: (Versi hanya untuk library yang penting, versi dependency mengikuti ketika instalasi):
- pytorch==1.8.0 (Untuk Jetson saat ini baru support 1.8.0)
- torchvision==0.9.0
- torchaudio==0.8.0
- opencv
- scikit-learn
- scikit-video
- pyyaml
- pandas
- tqdm
- pillow
- imutils
- imgaug
- albumentations
- requests
- torchviz
- torchsummary

### Setup Development Environment:
Contoh setup dengan menggunakan Anaconda/Miniconda (https://docs.conda.io/en/latest/miniconda.html).
Dikarenakan menggunakan Conda maka bisa dilakukan di semua versi Ubuntu.

- Create environment:
    ```bash
    conda create --name welding python=3.6
    conda activate welding
    ```

- Install library:

    Install PyTorch dan CUDA
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```

    Install library lain:
    ```bash
    pip install pytorch-lightning
    pip install albumentations
    pip install scikit-learn
    pip install scikit-video
    pip install pyyaml
    pip install pandas
    pip install tqdm
    pip install pillow
    pip install imutils
    pip install requests
    pip install torchviz
    pip install torchsummary
    ```

Notes: Bila ingin menggunakan TensorRT bisa dengan menggunakan nvidia-docker (https://github.com/NVIDIA/nvidia-docker) dengan image TensorRT (https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)

## How to Use:
### Dataset:
- Gunakan dataset Spot atau ARC dari Google Drive

### Preprocess Dataset:
- Step:
    1. `dataset_fix_missing.py` mengenerate mask untuk images yang label mask tidak lengkap
    2. `dataset_split.py` membagi dataset menjadi train dan validation
    3. `dataset_augmentation.py` mengenerate dataset hasil augmentasi
    4. `dataset_image_gradient.py` mengenerate dataset dengan image gradient sobel
    5. `dataset_lbp_image.py` mengenerate dataset dengan image LBP

### Training:
- `train.py` untuk melatih model
- Modifikasi `model_config.yml` untuk mengatur parameter training
- `train_pl.py` untuk melatih model (PyTorch-Lightning)

### Inference:
- Gunakan model yang sudah dilatih atau download model dari Google Drive
- `predict_img.py` untuk inference dengan input image
- `predict_vid.py` untuk inference dengan input video
- `predict_real.py` untuk inference dengan input kamera

- `predict_vid_pl.py` untuk inference dengan input video (PyTorch-Lightning)

- `predict_count_centroid.py` untuk inference + counting menggunakan centroid dengan input video
- `predict_count_sift.py` untuk inference + counting menggunakan feature descriptor SIFT dengan input video

- `predict_tensorrt.py` untuk inference menggunakan model yang dikuantisasi oleh tensorrt dengan input video

Notes: `predict_tensorrt*` require Nvidia GPU and TensorRT installed

### Additional:
- `evaluate_tensorrt.py` untuk mengecek kecepatan inference hasil kuantisasi model
- `evaluate_tensorrt_acc.py` untuk mengecek akurasi hasil kuantisasi model

### Tools:
- `tools/via_annotation_to_mask.py` untuk mengenerate file image mask dari data annotation hasil export VIA (VGG Image Annotator)
- `tools/labelbox_annotation_to_mask.py` untuk mengenerate file image mask dari data annotation hasil export labelbox
- `tools/labelbox_downloader.py` untuk mendownload image dan mengenerate file image mask dari data annotation hasil export labelbox
- `tools/ckpt_to_torchscript.py` untuk membuat torchscript dari checkpoint model
- `tools/dataset_resize.py` untuk meresize image di dataset (Membantu untuk mempercepat training)
- `tools/dataset_sampling.py` untuk membuat subset dari dataset
- `tools/ros_capture_image.py` untuk mencapture image ke folder `captured_image` berdasarkan waypoint di `example_wp.json`
- `tools/simple_stitch.py` untuk men-stiching di folder `captured_image` menjadi `result.jpg`
- `tools/stitch2D.py` 
untuk stitching frame yang ada di `image_path` dengan jumlah default `frame/stitch = 5` dan `resolution = 1024p`
nilai `frame/stitch` dan `resolution` bisa diatur argumen `--frames` dan `--res`

## Program Structure:
- `deeplearning/`
    - `CNNSegmentation.py`: Berisi definisi arsitektur neural network
- `postprocess/`
    - `counting_centroid` : Berisi fungsi untuk counting berbasis centroid
    - `counting_descriptor` : Berisi fungsi untuk counting berbasis feature descriptor
- `lightning_logs` : Berisi file log dan checkpoint training (PyTorch-Lightning)
- `model/` : Berisi model yang sudah dilatih (Download manual dari Google Drive)
- `dataset/` : Berisi dataset (Download manual dari Google Drive)
- `prediction/` : Berisi sample video (Download manual dari Google Drive)


## Run Tensorboard
- `tensorboard --logdir=/lightning_logs/`

## Reference:
- TODO
