import sys
import torch
from torch import cat, add, nn
import math

class VGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(mid_channels) #normalisasi dari sekian channel
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                    stride=1, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels) #normalisasi dari sekian channel
        

    def forward(self, x): #x adalah tensor yang dimasukkan
        x = self.conv1(x) #konvolusi
        x = self.bn1(x) #normalisasi
        x = self.relu(x) #aktivasi
        x = self.conv2(x) #konvolusi
        x = self.bn2(x) #normalisasi
        x = self.relu(x) #aktivasi
        #x = self.drop(x) #dropout
        return x

#baca: https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
class UNet(nn.Module):
    #default input channel adalah 3, asumsi pembacaan cv2 adalah RGB
    def __init__(self, n_class, conv_block, in_channel_dim=3, **kwargs):
        super().__init__()
        #jumlah channel feature map yang dikehendaki
        #alias jumlah kernel konvolusi pada setiap layernya
        n_fmap_ch = [32, 64, 128, 256, 512] 
        
        #pilih arsitektur Convolutional Block untuk downsampling
        down_Block = VGGBlock
        neck_Block = VGGBlock
        up_Block = VGGBlock
            
        #fungsi downsampling (dengan maxpooling) dan upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)     
        # Definisikan ukuran output channel secara manual
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
        #in_ch = 3
        x0_0 = self.conv0_0(input) #out_ch=32, HxW=inputHxW=256x256 
        down_x0_0 = self.pool(x0_0) #HxW = (256x256)/2 = 128x128
        x1_0 = self.conv1_0(down_x0_0)#in_ch = 32, out_ch = 64
        down_x1_0 = self.pool(x1_0) #HxW = (128x128)/2 = 64x64
        x2_0 = self.conv2_0(down_x1_0)#in_ch = 64, out_ch = 128
        down_x2_0 = self.pool(x2_0) #HxW = (64x64)/2 = 32x32
        x3_0 = self.conv3_0(down_x2_0) #in_ch = 128, out_ch = 256
        down_x3_0 = self.pool(x3_0) #HxW = (32x32)/2 = 16x16

        #bagian neck
        x4_0 = self.conv4_0(down_x3_0) #in_ch=256, out_ch=512
        
        #bagian upsampling
        #concatenate dengan setiap output di downsampling sebelumnya
        #concat pada axis dim 1, karena yang diconcate adalah channel feature mapnya
        #tensor dimension: batch x channel x H x W ->> 0,1,2,3, axis dim 1 adalah channel
        up_x4_0 = self.up1(x4_0) #HxW = (16x16)*2 = 32x32
        x3_1 = self.conv3_1(cat([x3_0, up_x4_0], dim=1)) #in_ch = 256+512 = 768, out_ch = 256,
        up_x3_1 = self.up2(x3_1) #HxW = (32x32)*2 = 32x64
        x2_2 = self.conv2_2(cat([x2_0, up_x3_1], dim=1)) #in_ch = 128+256 = 384, out_ch = 128, 
        up_x2_2 = self.up3(x2_2) #HxW = (64x64)*2 = 128x128
        x1_3 = self.conv1_3(cat([x1_0, up_x2_2], dim=1)) #in_ch = 64+128 = 192, out_ch = 64,
        up_x1_3 = self.up4(x1_3) #HxW = (128x128)*2 = 256x256
        x0_4 = self.conv0_4(cat([x0_0, up_x1_3], dim=1)) #in_ch = 32+64 = 96, out_ch = 32,
        
        #aktivasi menjadi output biner,
        #ukuran kernel konvolusi 1x1, sehingga ukuran HxW tidak berubah
        #in_ch=32, out_ch=jumlah_class=2, HxW=256x256 => [32,2,256x256]
        output = self.final(x0_4)
        return output

class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation=2)
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


def main():
    print("CHECK MODEL ARCHITECTURE")

#RUN PROGRAM
if __name__ == "__main__":
    main()
