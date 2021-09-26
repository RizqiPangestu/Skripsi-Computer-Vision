'''
>>>>>>>>>>> UPDATE V2.0 <<<<<<<<<<<
1. PENAMBAHAN ARSITEKTUR:
    - UNet
    - NestedUNet (new)
2. PENAMBAHAN FUNGSI CONV BLOCK:
    - VGG Block
    - Inception Block (new)
    - Residual Block (new)
    - Dense Block (new)
3. UPDATE CODE DI:
    - train.py
    - deeplearning.CNNSegmentation.py
    - predict_img.py
    - predict_vid.py
    
>>>>>>>>>>> UPDATE V2.1 <<<<<<<<<<<
1. PENAMBAHAN FUNGSI CONV BLOCK:
    - Squeeze Block

>>>>>>>>>>> UPDATE V3.0 <<<<<<<<<<<
1. MODIFIKASI ARSITEKTUR:
    - 1 ARSITEKTUR BISA MEMILIKI CONV BLOCK YANG BERBEDA UNTUK:
        - Backbone
        - Upsampling
        - Neck
2. PENAMBAHAN ARSITEKTUR:
    - DeepUNet
3. UPDATE CODE:
    - train.py
        - fungsi menyimpan index data train dan val
        - fungsi pilih load index atau split
'''

import sys
from torch import cat, add, nn
#YANG PERLU DICATAT ADALAH:
#JUMLAH CHANNEL / FEATURE MAP YANG DIHASILKAN SUATU LAYER
#ADALAH SAMA DENGAN JUMLAH KERNEL KONVOLUSI PADA LAYER TERSEBUT

#VGG Block, baca: https://d2l.ai/chapter_convolutional-modern/vgg.html
#tapi hanya 2x convolution saja, no maxpooling
#pooling dilakukan dibody UNet untuk mengecilkan dimensi tensor
class VGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        #fungsi2 yang diperlukan di VGGblock, hanya:
        #konvolusi, batch norm, dan aktivasi RELU
        self.relu = nn.ReLU(inplace=True) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv pertama
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(mid_channels) #normalisasi dari sekian channel
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml input channel conv kedua
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels) #normalisasi dari sekian channel
        #dropout module untuk mencegah overfitting # https://arxiv.org/abs/1207.0580
        #self.drop = nn.Dropout(p=0.5)

    #1 block berisi conv, batch normalisasi, dan aktivasi relu
    #VGGblock berisi 2 block tersebut
    def forward(self, x): #x adalah tensor yang dimasukkan
        x = self.conv1(x) #konvolusi
        x = self.bn1(x) #normalisasi
        x = self.relu(x) #aktivasi
        x = self.conv2(x) #konvolusi
        x = self.bn2(x) #normalisasi
        x = self.relu(x) #aktivasi
        #x = self.drop(x) #dropout
        #JADI 1 block (VGGblock) conv ini berisi = 2x(konvolusi-normalisasi-aktivasi)
        return x



#InceptionBlock, baca: https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
#versi sederhana, biar ga banyak parameter yang ditrain
#pooling dilakukan dibody UNet untuk mengecilkan dimensi tensor
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        #fungsi2 yang diperlukan di InceptionBlock, hanya:
        #konvolusi, batch norm, concat, dan aktivasi RELU
        self.relu = nn.ReLU(inplace=True) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.bn_mid = nn.BatchNorm2d(int(mid_channels/2)) #normalisasi dari sekian channel
        self.bn_out = nn.BatchNorm2d(out_channels) #normalisasi dari sekian channel
        #kernel filter konvolusi1 5x5
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 2 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv pertama
        self.conv1 = nn.Conv2d(in_channels, int(mid_channels/2), kernel_size=5,
                    stride=1, padding=2, padding_mode='zeros')
        #kernel filter konvolusi2 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv kedua
        self.conv2 = nn.Conv2d(in_channels, int(mid_channels/2),
                    kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        #kernel filter konvolusi3 1x1
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 0 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv ke-3
        self.conv3 = nn.Conv2d(in_channels, int(mid_channels/2),
                    kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        #kernel filter konvolusi4 3x3 (final)
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #3 x mid channel/2 sbg jml output channel conv ke-4
        self.conv4 = nn.Conv2d(int(3*mid_channels/2), out_channels,
                    kernel_size=3, stride=1, padding=1, padding_mode='zeros')		
        #dropout module untuk mencegah overfitting # https://arxiv.org/abs/1207.0580
        self.drop = nn.Dropout(p=0.5)
        
    #1 block berisi 3 mid conv, 1 final conv, batch normalisasi, dan aktivasi relu, dropout
    #InceptionBlock berisi 3 mid conv tersebut yang diconcat
    def forward(self, x0): #x adalah tensor yang dimasukkan
        #mid inception
        x_mid1 = self.conv1(x0) #konvolusi 5x5 dari x0
        x_mid1 = self.bn_mid(x_mid1) #normalisasi
        x_mid1 = self.relu(x_mid1) #aktivasi
        x_mid2 = self.conv2(x0) #konvolusi 3x3 dari x0
        x_mid2 = self.bn_mid(x_mid2) #normalisasi
        x_mid2 = self.relu(x_mid2) #aktivasi
        x_mid3 = self.conv3(x0) #konvolusi 1x1 dari x0
        x_mid3 = self.bn_mid(x_mid3) #normalisasi
        x_mid3 = self.relu(x_mid3) #aktivasi
        #final conv, concat dulu
        x_final = self.conv4(cat([x_mid1, x_mid2, x_mid3], dim=1)) #konvolusi dari x_mid1, x_mid2, x_mid3
        x_final = self.bn_out(x_final) #normalisasi
        x_final = self.relu(x_final) #aktivasi
        x_final = self.drop(x_final) #dropout
        #JADI 1 block (InceptionBlock) conv ini berisi = 3x(konvolusi-normalisasi-aktivasi) mid lalu diconcat bertumpuk
        #trus final conv diakhir
        return x_final
        


#Residual Block, baca: https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
#versi sederhana, biar ga banyak parameter yang ditrain
#pooling dilakukan dibody UNet untuk mengecilkan dimensi tensor
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        #fungsi2 yang diperlukan di ResBlock, hanya:
        #konvolusi, batch norm, add, dan aktivasi RELU
        self.relu = nn.ReLU(inplace=True) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv pertama
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(mid_channels) #normalisasi dari sekian channel
        #kernel filter konvolusi 1x1
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 0 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #in channel sbg input channel conv2 karena mengambil dari input channel langsung
        #outputnya sejumlah mid channel karena dijumlahkan dengan output conv1
        self.conv2 = nn.Conv2d(in_channels, mid_channels,
                    kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.bn2 = self.bn1 #normalisasi bn2=bn1 karena akan dijumlahkan
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml input channel conv terakhir
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                    kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn3 = nn.BatchNorm2d(out_channels) #normalisasi dari sekian channel
        #dropout module untuk mencegah overfitting # https://arxiv.org/abs/1207.0580
        self.drop = nn.Dropout(p=0.5)

    #1 block berisi conv, batch normalisasi, dan aktivasi relu
    #ResBlock berisi 2 block tersebut yang diconcat bertumpuk + 1 konvolusi akhir, dropout
    def forward(self, x0): #x adalah tensor yang dimasukkan
        x1 = self.conv1(x0) #konvolusi dari x0
        x1 = self.bn1(x1) #normalisasi
        x1 = self.relu(x1) #aktivasi
        x2 = self.conv2(x0) #konvolusi dari x0 untuk mendapatkan fmap yang sama dengan x1
        #x2 = self.bn2(x2) #normalisasi
        #x2 = self.relu(x2) #aktivasi
        x3 = self.conv3(add(x1, x2)) #konvolusi dari x1 + x2
        x3 = self.bn3(x3) #normalisasi
        x3 = self.relu(x3) #aktivasi
        x3 = self.drop(x3) #dropout
        #JADI 1 block (ResBlock) conv ini berisi = 2x(konvolusi-normalisasi-aktivasi) yang dijumlah + 1 konvolusi akhir
        return x3



#Dense Block, baca: https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a
#versi sederhana, biar ga banyak parameter yang ditrain
#pooling dilakukan dibody UNet untuk mengecilkan dimensi tensor
class DenseBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        #fungsi2 yang diperlukan di DenseBlock, hanya:
        #konvolusi, batch norm, concat, dan aktivasi RELU
        self.relu = nn.ReLU(inplace=True) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv pertama
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(mid_channels) #normalisasi dari sekian channel
        #kernel filter konvolusi 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel+in channel sbg jml input channel conv kedua
        self.conv2 = nn.Conv2d((in_channels+mid_channels), out_channels,
                    kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels) #normalisasi dari sekian channel
        #ini transision layer, konvolusi 1x1
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 0 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #input channel adalah in_mid_out yang diconcat jadi 1
        self.conv3 = nn.Conv2d((in_channels+mid_channels+out_channels), out_channels,
                    kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.bn3 = self.bn2 #jumlah channel yang dinomralisasi sama dengan bn2
        #dropout module untuk mencegah overfitting # https://arxiv.org/abs/1207.0580
        self.drop = nn.Dropout(p=0.5)

    #1 block berisi conv, batch normalisasi, dan aktivasi relu
    #DenseBlock berisi 2 block tersebut yang diconcat bertumpuk + 1 transisition layer, dropout
    def forward(self, x0): #x adalah tensor yang dimasukkan
        x1 = self.conv1(x0) #konvolusi dari x0
        x1 = self.bn1(x1) #normalisasi
        x1 = self.relu(x1) #aktivasi
        x2 = self.conv2(cat([x1, x0], dim=1)) #konvolusi dari x0_x1
        x2 = self.bn2(x2) #normalisasi
        x2 = self.relu(x2) #aktivasi
        x3 = self.conv3(cat([x2, x1, x0], dim=1)) #konvolusi dari x0_x1_x2
        x3 = self.bn3(x3) #normalisasi
        x3 = self.relu(x3) #aktivasi
        x3 = self.drop(x3) #dropout
        #JADI 1 block (DenseBlock) conv ini berisi = 2x(konvolusi-normalisasi-aktivasi) yang diconcat bertumpuk + 1 transition layer 1x1
        return x3



#Fire Modul pada SqueezeNet, baca: https://codelabs.developers.google.com/codelabs/keras-flowers-squeezenet/#6
#ambil fire modul saja untuk squeeze dan expand
#pooling dilakukan dibody UNet untuk mengecilkan dimensi tensor
class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        #fungsi2 yang diperlukan di SqueezeBlock, hanya:
        #konvolusi, batch norm, concat, dan aktivasi RELU
        self.relu = nn.ReLU(inplace=True) # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.bn_s = nn.BatchNorm2d(mid_channels) #normalisasi
        self.bn_e = nn.BatchNorm2d(int(out_channels/2)) #normalisasi
        #kernel filter konvolusi1 untuk squeeze (1x1)
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 0 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #mid channel sbg jml output channel conv pertama (squeeze)
        self.s_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                    stride=1, padding=0, padding_mode='zeros')
        #kernel filter konvolusi2 (expand 1) 3x3
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 1 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #output channel dibagi 2 karena nanti akan diconcat dengan conv expand kedua lagi
        self.e_conv3 = nn.Conv2d(mid_channels, int(out_channels/2),
                    kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        #kernel filter konvolusi2 (expand 2) 1x1
        #stride pergeseran kernel tiap 1 pixel
        #tambah padding 0 untuk H dan W, padding = 0
        #sehingga output H x W tidak berubah setelah proses convolusi
        #output channel dibagi 2 karena nanti akan diconcat dengan conv expand pertama lagi
        self.e_conv1 = nn.Conv2d(mid_channels, int(out_channels/2),
                    kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        #dropout module untuk mencegah overfitting # https://arxiv.org/abs/1207.0580
        #self.drop = nn.Dropout(p=0.5)
        
        
    #1 block berisi 1 squeeze conv, 2 expand conv, batch normalisasi, dan aktivasi relu
    #expand conv outputnya di concat diakhir
    def forward(self, x0): #x adalah tensor yang dimasukkan
        #mid inception
        s1 = self.s_conv1(x0) #konvolusi squeeze 1x1 dari x0
        s1 = self.bn_s(s1) #normalisassi
        s1 = self.relu(s1) #aktivasi
        e3 = self.e_conv3(s1) #konvolusi expand 3x3 dari s1
        e3 = self.bn_e(e3) #normalisassi
        e3 = self.relu(e3) #aktivasi
        e1 = self.e_conv1(s1) #konvolusi expand 1x1 dari s1
        e1 = self.bn_e(e1) #normalisassi
        e1 = self.relu(e1) #aktivasi
        #concat diakhir
        y = cat([e3, e1], dim=1) #concat e1 dan e3 pada axis channelnya
        #y = self.drop(y) #dropout
        #JADI 1 block (SqueezeBlock) = 1 squeeze conv, 2 expand conv, batch normalisasi, dan aktivasi relu
        #expand conv outputnya di concat diakhir
        return y

#baca: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
class UNet(nn.Module): # https://arxiv.org/pdf/1505.04597.pdf
    #default input channel adalah 3, asumsi pembacaan cv2 adalah RGB
    def __init__(self, n_class, conv_block, in_channel_dim=3, **kwargs):
        super().__init__()
        #jumlah channel feature map yang dikehendaki
        #alias jumlah kernel konvolusi pada setiap layernya
        n_fmap_ch = [32, 64, 128, 256, 512] 
        
        #pilih arsitektur Convolutional Block untuk downsampling
        if conv_block[0]=="VGG":
            down_Block = VGGBlock
        elif conv_block[0]=="Inception":
            down_Block = InceptionBlock
        elif conv_block[0]=="Residual":
            down_Block = ResBlock
        elif conv_block[0]=="Dense":
            down_Block = DenseBlock
        elif conv_block[0]=="Squeeze":
            down_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk neck
        if conv_block[1]=="VGG":
            neck_Block = VGGBlock
        elif conv_block[1]=="Inception":
            neck_Block = InceptionBlock
        elif conv_block[1]=="Residual":
            neck_Block = ResBlock
        elif conv_block[1]=="Dense":
            neck_Block = DenseBlock
        elif conv_block[1]=="Squeeze":
            neck_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk upsampling
        if conv_block[2]=="VGG":
            up_Block = VGGBlock
        elif conv_block[2]=="Inception":
            up_Block = InceptionBlock
        elif conv_block[2]=="Residual":
            up_Block = ResBlock
        elif conv_block[2]=="Dense":
            up_Block = DenseBlock
        elif conv_block[2]=="Squeeze":
            up_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
            
        #fungsi downsampling (dengan maxpooling) dan upsampling
        #kernel pooling HxW = 2x2, no padding dan
        #stride=2 sehingga dimensi HxW ter-downsampling menjadi H/2 x W/2
        #max pooling, berarti dari 2x2 kotak pixel diambil yang paling besar (max)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        #spatial (HxW) size dikali oleh scale_factor=2
        #sehingga HxW ter-upsampling menjadi H*2 x W*2
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Definisikan ukuran output layer secara manual
        self.up1 = nn.Upsample(size=[32,32], mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(size=[64,64], mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(size=[128,128], mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(size=[256,256], mode='bilinear', align_corners=True)
        
        #bagian downsampling
        # format seperti Block di atas: in_channels, mid_channels, out_channels
        # isi block tergantung conv block yang dipakai, lihat di atas
        self.conv0_0 = down_Block(in_channel_dim, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0 = down_Block(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0 = down_Block(n_fmap_ch[1], n_fmap_ch[2], n_fmap_ch[2])
        self.conv3_0 = down_Block(n_fmap_ch[2], n_fmap_ch[3], n_fmap_ch[3])
        #bagian neck
        self.conv4_0 = neck_Block(n_fmap_ch[3], n_fmap_ch[4], n_fmap_ch[4])
        
        #bagian upsampling
        #jumlahkan channel output layer sebelumnya dengan channel output pada downsampling yang sesuai
        # isi block tergantung conv block yang dipakai, lihat di atas
        self.conv3_1 = up_Block(n_fmap_ch[3]+n_fmap_ch[4], n_fmap_ch[3], n_fmap_ch[3])
        self.conv2_2 = up_Block(n_fmap_ch[2]+n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_3 = up_Block(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_4 = up_Block(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        
        #n_class sebagai channel output akhir
        #1 x konvolusi yang menghasilkan sejumlah n_class output feature map
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        self.final = nn.Conv2d(n_fmap_ch[0], n_class, kernel_size=1)

    def forward(self, input):
        #perhatikan n_fmap_ch = [32, 64, 128, 256, 512] di atas
        #bagian downsampling
        x0_0 = self.conv0_0(input) #in_ch=in_channel_dim=RGB=3, out_ch=32, HxW=inputHxW=128x256 
        down_x0_0 = self.pool(x0_0) #HxW=(128x256)/2=64x128
        x1_0 = self.conv1_0(down_x0_0)#in_ch=32, out_ch=64
        down_x1_0 = self.pool(x1_0) #HxW=(64x128)/2=32x64
        x2_0 = self.conv2_0(down_x1_0)#in_ch=64, out_ch=128
        down_x2_0 = self.pool(x2_0) #HxW=(32x64)/2=16x32
        x3_0 = self.conv3_0(down_x2_0) #in_ch=128, out_ch=256
        down_x3_0 = self.pool(x3_0) #HxW=(16x32)/2=8x16
        #bagian neck
        x4_0 = self.conv4_0(down_x3_0) #in_ch=256, out_ch=512
        
        #bagian upsampling
        #dan concatenate dengan setiap output di downsampling sebelumnya
        #concat pada axis dim 1, karena yang diconcate adalah channel feature mapnya
        #ingat! tensor dimension: batch x channel x H x W ->> 0,1,2,3, axis dim 1 adalah channel
        up_x4_0 = self.up1(x4_0) #HxW=(8x16)*2=16x32
        x3_1 = self.conv3_1(cat([x3_0, up_x4_0], dim=1)) #in_ch=256+512=768, out_ch=256, 
        up_x3_1 = self.up2(x3_1) #HxW=(16x32)*2=32x64
        x2_2 = self.conv2_2(cat([x2_0, up_x3_1], dim=1)) #in_ch=128+256=384, out_ch=128, 
        up_x2_2 = self.up3(x2_2) #HxW=(32x64)*2=64x128
        x1_3 = self.conv1_3(cat([x1_0, up_x2_2], dim=1)) #in_ch=64+128=192, out_ch=64,
        up_x1_3 = self.up4(x1_3) #HxW=(64x128)*2=128x256
        x0_4 = self.conv0_4(cat([x0_0, up_x1_3], dim=1)) #in_ch=32+64=96, out_ch=32,
        
        #perhatikan self_final,
        #jika hanya 1 class maka output UNet ini juga 1 lapis layer saja
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        #in_ch=32, out_ch=jumlah_class=1, HxW=128x256
        output = self.final(x0_4)
        return output


# baca: https://medium.com/@sh.tsang/review-unet-a-nested-u-net-architecture-biomedical-image-segmentation-57be56859b20
class NestedUNet(nn.Module): #aka UNet++ https://arxiv.org/pdf/1807.10165.pdf
    #default input channel adalah 3, asumsi pembacaan cv2 adalah RGB
    def __init__(self, n_class, conv_block, in_channel_dim=3, **kwargs):
        super().__init__()
        #jumlah channel feature map yang dikehendaki
        #alias jumlah kernel konvolusi pada setiap layernya
        n_fmap_ch = [32, 64, 128, 256, 512]
        
        #pilih arsitektur Convolutional Block untuk downsampling
        if conv_block[0]=="VGG":
            down_Block = VGGBlock
        elif conv_block[0]=="Inception":
            down_Block = InceptionBlock
        elif conv_block[0]=="Residual":
            down_Block = ResBlock
        elif conv_block[0]=="Dense":
            down_Block = DenseBlock
        elif conv_block[0]=="Squeeze":
            down_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk neck
        if conv_block[1]=="VGG":
            neck_Block = VGGBlock
        elif conv_block[1]=="Inception":
            neck_Block = InceptionBlock
        elif conv_block[1]=="Residual":
            neck_Block = ResBlock
        elif conv_block[1]=="Dense":
            neck_Block = DenseBlock
        elif conv_block[1]=="Squeeze":
            neck_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk upsampling
        if conv_block[2]=="VGG":
            up_Block = VGGBlock
        elif conv_block[2]=="Inception":
            up_Block = InceptionBlock
        elif conv_block[2]=="Residual":
            up_Block = ResBlock
        elif conv_block[2]=="Dense":
            up_Block = DenseBlock
        elif conv_block[2]=="Squeeze":
            up_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #fungsi downsampling (dengan maxpooling) dan upsampling
        #kernel pooling HxW = 2x2, no padding dan
        #stride=2 sehingga dimensi HxW ter-downsampling menjadi H/2 x W/2
        #max pooling, berarti dari 2x2 kotak pixel diambil yang paling besar (max)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        #spatial (HxW) size dikali oleh scale_factor=2
        #sehingga HxW ter-upsampling menjadi H*2 x W*2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #supaya lebih paham, baca ini:
        # https://medium.com/@sh.tsang/review-unet-a-nested-u-net-architecture-biomedical-image-segmentation-57be56859b20
        #bagian downsampling
        self.conv0_0 = down_Block(in_channel_dim, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0 = down_Block(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0 = down_Block(n_fmap_ch[1], n_fmap_ch[2], n_fmap_ch[2])
        self.conv3_0 = down_Block(n_fmap_ch[2], n_fmap_ch[3], n_fmap_ch[3])
        #bagian neck
        self.conv4_0 = neck_Block(n_fmap_ch[3], n_fmap_ch[4], n_fmap_ch[4])
        #bagian upsampling 1
        self.conv0_1 = up_Block(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_1 = up_Block(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_1 = up_Block(n_fmap_ch[2]+n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv3_1 = up_Block(n_fmap_ch[3]+n_fmap_ch[4], n_fmap_ch[3], n_fmap_ch[3])
        #bagian upsampling 2
        self.conv0_2 = up_Block(n_fmap_ch[0]*2+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_2 = up_Block(n_fmap_ch[1]*2+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_2 = up_Block(n_fmap_ch[2]*2+n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        #bagian upsampling 3
        self.conv0_3 = up_Block(n_fmap_ch[0]*3+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_3 = up_Block(n_fmap_ch[1]*3+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        #bagian upsampling 4
        self.conv0_4 = up_Block(n_fmap_ch[0]*4+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        
        #n_class sebagai channel output akhir
        #1 x konvolusi yang menghasilkan sejumlah n_class output feature map
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        self.final = nn.Conv2d(n_fmap_ch[0], n_class, kernel_size=1)

    #supaya lebih paham, baca ini:
    # https://medium.com/@sh.tsang/review-unet-a-nested-u-net-architecture-biomedical-image-segmentation-57be56859b20
    #perhatikan tanda panahnya
    def forward(self, input):
        #concat pada axis dim 1, karena yang diconcate adalah channel feature mapnya
        #ingat! tensor dimension: batch x channel x H x W ->> 0,1,2,3, axis dim 1 adalah channel
        
        #stage 0
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(cat([x0_0, self.up(x1_0)], dim=1))
        #stage 1
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        #stage 2
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        #stage 3 paling kanan
        x4_0 = self.conv4_0(self.pool(x3_0)) #neck
        x3_1 = self.conv3_1(cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
        
        #perhatikan self_final,
        #jika hanya 1 class maka output UNet ini juga 1 lapis layer saja
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        #in_ch=32, out_ch=jumlah_class=1, HxW=128x256
        output = self.final(x0_4)
        return output

#Deeper UNet sama kayak UNet di atas tetapi dengan arsitektur yang lebih big (tambah 2 block)
#baca: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
class DeepUNet(nn.Module): # https://arxiv.org/pdf/1505.04597.pdf
    #default input channel adalah 3, asumsi pembacaan cv2 adalah RGB
    def __init__(self, n_class, conv_block, in_channel_dim=3, **kwargs):
        super().__init__()
        #jumlah channel feature map yang dikehendaki
        #alias jumlah kernel konvolusi pada setiap layernya
        n_fmap_ch = [8, 16, 32, 64, 128, 256, 512] 
        
        #pilih arsitektur Convolutional Block untuk downsampling
        if conv_block[0]=="VGG":
            down_Block = VGGBlock
        elif conv_block[0]=="Inception":
            down_Block = InceptionBlock
        elif conv_block[0]=="Residual":
            down_Block = ResBlock
        elif conv_block[0]=="Dense":
            down_Block = DenseBlock
        elif conv_block[0]=="Squeeze":
            down_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk neck
        if conv_block[1]=="VGG":
            neck_Block = VGGBlock
        elif conv_block[1]=="Inception":
            neck_Block = InceptionBlock
        elif conv_block[1]=="Residual":
            neck_Block = ResBlock
        elif conv_block[1]=="Dense":
            neck_Block = DenseBlock
        elif conv_block[1]=="Squeeze":
            neck_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
        
        #pilih arsitektur Convolutional Block untuk upsampling
        if conv_block[2]=="VGG":
            up_Block = VGGBlock
        elif conv_block[2]=="Inception":
            up_Block = InceptionBlock
        elif conv_block[2]=="Residual":
            up_Block = ResBlock
        elif conv_block[2]=="Dense":
            up_Block = DenseBlock
        elif conv_block[2]=="Squeeze":
            up_Block = SqueezeBlock
        else:
            sys.exit("ERROR, CONV BLOCK TIDAK ADA....................")
            
        #fungsi downsampling (dengan maxpooling) dan upsampling
        #kernel pooling HxW = 2x2, no padding dan
        #stride=2 sehingga dimensi HxW ter-downsampling menjadi H/2 x W/2
        #max pooling, berarti dari 2x2 kotak pixel diambil yang paling besar (max)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        #spatial (HxW) size dikali oleh scale_factor=2
        #sehingga HxW ter-upsampling menjadi H*2 x W*2
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #bagian downsampling
        # format seperti Block di atas: in_channels, mid_channels, out_channels
        # isi block tergantung conv block yang dipakai, lihat di atas
        self.conv0_0 = down_Block(in_channel_dim, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0 = down_Block(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0 = down_Block(n_fmap_ch[1], n_fmap_ch[2], n_fmap_ch[2])
        self.conv3_0 = down_Block(n_fmap_ch[2], n_fmap_ch[3], n_fmap_ch[3])
        self.conv4_0 = down_Block(n_fmap_ch[3], n_fmap_ch[4], n_fmap_ch[4])
        #tambahan 2 block Bigger
        self.conv5_0 = down_Block(n_fmap_ch[4], n_fmap_ch[5], n_fmap_ch[5])
        #bagian neck
        self.conv6_0 = neck_Block(n_fmap_ch[5], n_fmap_ch[6], n_fmap_ch[6])
        
        #bagian upsampling
        #jumlahkan channel output layer sebelumnya dengan channel output pada downsampling yang sesuai
        # isi block tergantung conv block yang dipakai, lihat di atas
        #tambahan 2 block Bigger
        self.conv5_1 = up_Block(n_fmap_ch[5]+n_fmap_ch[6], n_fmap_ch[5], n_fmap_ch[5])
        self.conv4_2 = up_Block(n_fmap_ch[4]+n_fmap_ch[5], n_fmap_ch[4], n_fmap_ch[4])
        #block asli
        self.conv3_3 = up_Block(n_fmap_ch[3]+n_fmap_ch[4], n_fmap_ch[3], n_fmap_ch[3])
        self.conv2_4 = up_Block(n_fmap_ch[2]+n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_5 = up_Block(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_6 = up_Block(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        
        #n_class sebagai channel output akhir
        #1 x konvolusi yang menghasilkan sejumlah n_class output feature map
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        self.final = nn.Conv2d(n_fmap_ch[0], n_class, kernel_size=1)

    def forward(self, input):
        #perhatikan n_fmap_ch = [8, 16, 32, 64, 128, 256, 512]  di atas
        #bagian downsampling
        x0_0 = self.conv0_0(input) #in_ch=in_channel_dim=RGB=3, out_ch=8, HxW=inputHxW=128x256 
        down_x0_0 = self.pool(x0_0) #HxW=(128x256)/2=64x128
        x1_0 = self.conv1_0(down_x0_0)#in_ch=8, out_ch=16
        down_x1_0 = self.pool(x1_0) #HxW=(64x128)/2=32x64
        x2_0 = self.conv2_0(down_x1_0)#in_ch=16, out_ch=32
        down_x2_0 = self.pool(x2_0) #HxW=(32x64)/2=16x32
        x3_0 = self.conv3_0(down_x2_0) #in_ch=32, out_ch=64
        down_x3_0 = self.pool(x3_0) #HxW=(16x32)/2=8x16
        x4_0 = self.conv4_0(down_x3_0) #in_ch=64, out_ch=128
        #Bigger
        down_x4_0 = self.pool(x4_0) #HxW=(8x16)/2=4x8
        x5_0 = self.conv5_0(down_x4_0) #in_ch=128, out_ch=256
        down_x5_0 = self.pool(x5_0) #HxW=(4x8)/2=2x4
        #bagian neck
        x6_0 = self.conv6_0(down_x5_0) #in_ch=256, out_ch=512
        
        #bagian upsampling
        #dan concatenate dengan setiap output di downsampling sebelumnya
        #concat pada axis dim 1, karena yang diconcate adalah channel feature mapnya
        #ingat! tensor dimension: batch x channel x H x W ->> 0,1,2,3, axis dim 1 adalah channel
        #Bigger
        up_x6_0 = self.up(x6_0) #HxW=(2x4)*2=4x8
        x5_1 = self.conv5_1(cat([x5_0, up_x6_0], dim=1)) #in_ch=256+512=768, out_ch=256, 
        up_x5_1 = self.up(x5_1) #HxW=(4x8)*2=8x16
        x4_2 = self.conv4_2(cat([x4_0, up_x5_1], dim=1)) #in_ch=128+256=384, out_ch=128, 
        #asli
        up_x4_2 = self.up(x4_2) #HxW=(8x16)*2=16x32
        x3_3 = self.conv3_3(cat([x3_0, up_x4_2], dim=1)) #in_ch=64+128=192, out_ch=64, 
        up_x3_3 = self.up(x3_3) #HxW=(16x32)*2=32x64
        x2_4 = self.conv2_4(cat([x2_0, up_x3_3], dim=1)) #in_ch=32+64=96, out_ch=32, 
        up_x2_4 = self.up(x2_4) #HxW=(32x64)*2=64x128
        x1_5 = self.conv1_5(cat([x1_0, up_x2_4], dim=1)) #in_ch=16+32=192, out_ch=16,
        up_x1_5 = self.up(x1_5) #HxW=(64x128)*2=128x256
        x0_6 = self.conv0_6(cat([x0_0, up_x1_5], dim=1)) #in_ch=8+16=96, out_ch=8,
        
        #perhatikan self_final,
        #jika hanya 1 class maka output UNet ini juga 1 lapis layer saja
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        #in_ch=32, out_ch=jumlah_class=1, HxW=128x256
        output = self.final(x0_6)
        return output

def main():
    print("CHECK MODEL ARCHITECTURE")

#RUN PROGRAM
if __name__ == "__main__":
    main()
