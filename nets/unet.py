import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from nets.dilatedblock import DilatedBlock, CBAM, NCHWtoNHWC, NHWCtoNCHW, GRNwithNHWC, DropPath, get_bn, trunc_normal_
from nets.dmamba import MambaLayer
# from dilatedblock import DilatedBlock, CBAM, NCHWtoNHWC, NHWCtoNCHW, GRNwithNHWC, DropPath, get_bn, trunc_normal_
# from dmamba import MambaLayer
from fvcore.nn import FlopCountAnalysis


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise_conv(x)
        return x 
    
class MultiStrideWindmillConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, strides, padding=1, dilation=1, bias=True):
        """
        strides: A list or tuple of 4 elements, representing the stride for each of the 4 convolutions.
        """
        super(MultiStrideWindmillConv2d, self).__init__()
        # 确保strides是一个四元素的列表或元组
        assert len(strides) == 4, "strides must be a list or tuple of 4 elements."
        
        self.conv1 = SeparableConv2d(in_channels, out_channels, strides[0], padding, dilation, bias)
        self.conv2 = SeparableConv2d(in_channels, out_channels, strides[1], padding, dilation, bias)
        self.conv3 = SeparableConv2d(in_channels, out_channels, strides[2], padding, dilation, bias)
        self.conv4 = SeparableConv2d(in_channels, out_channels, strides[3], padding, dilation, bias)
        
        # 添加深度卷积以融合特征，注意这里的输入通道数需要根据前面卷积的输出调整
        self.depthwise_conv = nn.Conv2d(out_channels*4, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        # print("AAAAAAAAAA:", x.shape)
        x1 = self.conv1(x)
        x2 = torch.flip(self.conv2(torch.flip(x, [3])), [3])
        x3 = torch.flip(torch.flip(self.conv3(torch.flip(torch.flip(x, [2]), [3])), [3]), [2])
        x4 = torch.flip(self.conv4(torch.flip(x, [2])), [2])
        # print("AAAAAAAAAA:", x4.shape)
        x = torch.cat((x1, x2, x3, x4), 1)
        # print("bbbbbbbbbb:", x.shape)
        x = self.depthwise_conv(x)
        # print("cccccccccc:", x.shape)
        return x

class EncDilatedBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 kernel_size,
                 ffn_factor = 4,
                 layer_scale_init_value=1e-6,
                 drop_path = 0,
                 with_cp = False,
                 deploy = False,
                 use_sync_bn = True,
                 attempt_use_lk_impl = True):
        super(EncDilatedBlock, self).__init__()
        self.dwconv = DilatedBlock(in_channels, 
                                   kernel_size,
                                   deploy=deploy,          
                                   use_sync_bn=use_sync_bn,
                                   attempt_use_lk_impl=attempt_use_lk_impl)
        ffn_dim = int(ffn_factor * in_channels)
        
        self.with_cp = with_cp
        
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')
            
        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(in_channels, use_sync_bn=use_sync_bn)
        self.cam = CBAM(in_channels)
        
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(in_channels, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, in_channels),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, in_channels, bias=False),
                NHWCtoNCHW(),
                get_bn(in_channels, use_sync_bn=use_sync_bn))
            
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(in_channels),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def compute_residual(self, x):
        # print("aaaaaaaaaaaaa:", x.shape)
        # print("aaaaaaaaaaaaa:", self.norm(self.dwconv(x)).shape)
        y = self.cam(self.norm(self.dwconv(x)))
        # y = self.norm(self.dwconv(x))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)
    
    def forward(self, inputs):

        def _f(x):
            return x + self.compute_residual(x)

        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out
    
class ResidualExtract(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 ffn_factor = 4,
                 layer_scale_init_value=1e-6,
                 drop_path = 0,
                 with_cp = False,
                 deploy = False,
                 use_sync_bn = True,
                 attempt_use_lk_impl = True):
        super(ResidualExtract, self).__init__()
        self.encitem1 = EncDilatedBlock(in_channels,
                                      kernel_size,
                                      ffn_factor = 4,
                                      layer_scale_init_value=1e-6,
                                      drop_path = 0,
                                      with_cp = False,
                                      deploy = False,
                                      use_sync_bn = True,
                                      attempt_use_lk_impl = True)
        self.encitem2 = EncDilatedBlock(in_channels,
                                      kernel_size,
                                      ffn_factor = 4,
                                      layer_scale_init_value=1e-6,
                                      drop_path = 0,
                                      with_cp = False,
                                      deploy = False,
                                      use_sync_bn = True,
                                      attempt_use_lk_impl = True)
        self.conv = SeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.encitem1(x)  # 1, 4, 160, 160 --> 1, 4, 160, 160
        z = self.encitem2(y)  # 1, 4 ,160, 160 --> 1, 4, 160, 160
        z = self.relu(self.bn(self.conv(z + x)))
        return z
           
class UMambaEnc_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 ffn_factor = 4,
                 layer_scale_init_value=1e-6,
                 drop_path = 0,
                 with_cp = False,
                 deploy = False,
                 use_sync_bn = True,
                 attempt_use_lk_impl = True):
        super(UMambaEnc_block, self).__init__()
        self.residual = ResidualExtract(in_channels, 
                                        out_channels,
                                        kernel_size,
                                        deploy = False,
                                        use_sync_bn = True,
                                        attempt_use_lk_impl = True)
        self.mamba = MambaLayer(out_channels)
        

    def forward(self, inputs):
        inputs = self.residual(inputs)  # 1, 4, 160, 160
        y = self.mamba(inputs)         
        x = F.max_pool2d(y, 2, stride=2)
        return x, y

class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = SeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class UDMamba(nn.Module):
    def __init__(self, *args):
        in_chan = 4
        out_chan = 3
        super(UDMamba, self).__init__()
        self.stride = [1,1,1,1]
        self.down1 = UMambaEnc_block(in_chan, 64, 13)
        self.down2 = UMambaEnc_block(64, 128, 13)
        self.down3 = UMambaEnc_block(128, 256, 13)
        self.down4 = UMambaEnc_block(256, 512, 13)
        self.conv1 = SeparableConv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = SeparableConv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)
        self.outconvp1 = nn.Conv2d(64, out_chan, 1)
        self.outconvm1 = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.bn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return x1
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = UDMamba()
    unet.to(device)
    a = torch.rand(1, 4, 160, 160).cuda()
    flop_counter = FlopCountAnalysis(unet, a)
    print(flop_counter.total())
    b = unet(a)
    print(b.shape)