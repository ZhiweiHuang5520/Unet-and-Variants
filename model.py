import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.resnet as resnet
import math


class ResBlock(nn.Module ):
    def __init__(self, inch, outch, stride=1, dilation=1 ):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of  convolution layer
        super(ResBlock, self ).__init__()
        assert(stride == 1 or stride == 2 )

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding = dilation, bias=False,
                dilation = dilation )
        self.bn1 = nn.BatchNorm2d(outch )
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding = dilation, bias=False,
                dilation = dilation )
        self.bn2 = nn.BatchNorm2d(outch )

        if inch != outch:
            self.mapping = nn.Sequential(
                        nn.Conv2d(inch, outch, 1, stride, bias=False ),
                        nn.BatchNorm2d(outch )
                    )
        else:
            self.mapping = None

    def forward(self, x ):
        y = x
        if not self.mapping is None:
            y = self.mapping(y )

        out = F.relu(self.bn1(self.conv1(x) ), inplace=True )
        out = self.bn2(self.conv2(out ) )

        out += y
        out = F.relu(out, inplace=True )

        return out


class encoder(nn.Module ):
    def __init__(self ):
        super(encoder, self ).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 2)
        self.b3_2 = ResBlock(256, 256, 1)

        self.b4_1 = ResBlock(256, 512, 2)
        self.b4_2 = ResBlock(512, 512, 1)

    def forward(self, im ):
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        return x1, x2, x3, x4, x5


class decoder(nn.Module ):
    def __init__(self ):
        super(decoder, self).__init__()
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )

    def forward(self, im, x1, x2, x3, x4, x5):

        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )

        return pred



class encoderDilation(nn.Module ):
    def __init__(self ):
        super(encoderDilation, self ).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)
    def forward(self, im ):

        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.b4_2(self.b4_1(x4 ) )
        return x1, x2, x3, x4, x5


class decoderDilation(nn.Module ):
    def __init__(self, isSpp = False ):
        super(decoderDilation, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(512+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )
    def forward(self, im, x1, x2, x3, x4, x5):

        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )

        return pred

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def upsample(input, size=None, scale_factor=None, align_corners=False):
    out = F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
    return out


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_size = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[0]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[1]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[2]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[3]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.pool1(x)
        out1 = upsample(out1, size=x.size()[-2:])

        out2 = self.pool2(x)
        out2 = upsample(out2, size=x.size()[-2:])

        out3 = self.pool3(x)
        out3 = upsample(out3, size=x.size()[-2:])

        out4 = self.pool4(x)
        out4 = upsample(out4, size=x.size()[-2:])

        out = torch.cat([x, out1, out2, out3, out4], dim=1)

        return out

class encoderSPP(nn.Module ):
    def __init__(self ):
        super(encoderSPP, self ).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)
        self.spp = PyramidPooling(512)
        
    def forward(self, im ):

        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im) ), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1 ) ) )
        x3 = self.b2_2(self.b2_1(x2 ) )
        x4 = self.b3_2(self.b3_1(x3 ) )
        x5 = self.spp(self.b4_2(self.b4_1(x4 ) ))
        return x1, x2, x3, x4, x5


class decoderSPP(nn.Module ):
    def __init__(self, isSpp = False ):
        super(decoderSPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(1024+256+128, 512, 3, 1, 1, bias=False )
        self.bn1 = nn.BatchNorm2d(512 )
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False )
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64+21, 21, 3, 1, 1, bias=False )
        self.bn2 = nn.BatchNorm2d(21 )
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.bn3 = nn.BatchNorm2d(21 )
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False )
        self.sf = nn.Softmax(dim=1 )
    def forward(self, im, x1, x2, x3, x4, x5):

        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat( [x3, x4, x5], dim=1) ) ), inplace=True )
        y1 = F.relu(self.bn1_1(self.conv1_1(y1 ) ), inplace = True )

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1) ), inplace=True )

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear' )
        y3 = F.relu(self.bn3(self.conv3(y2) ), inplace=True )

        y4 = self.sf(self.conv4(y3 ) )

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8) )
        
        return pred


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # upsample.shape = torch.Size([8, 3, 32, 32])
        image_features = F.interpolate(image_features, size=size, mode='bilinear')

        # block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.atrous_block1(x)

        # block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.atrous_block6(x)

        # block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.atrous_block12(x)

        # block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.atrous_block18(x)

        # torch.cat.shape = torch.Size([8, 15, 32, 32])
        # conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class encoderASPP(nn.Module):
    def __init__(self):
        super(encoderASPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)
        self.aspp = ASPP(512,1024)

    def forward(self, im):
        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.aspp(self.b4_2(self.b4_1(x4)))
        return x1, x2, x3, x4, x5


class decoderASPP(nn.Module):
    def __init__(self, isSpp=False):
        super(decoderASPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(1024 + 256 + 128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64 + 21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(21)
        self.conv4 = nn.Conv2d(21, 21, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):
        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear')
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        pred = -torch.log(torch.clamp(y4, min=1e-8))

        return pred



def loadPretrainedWeight(network, isOutput = False ):
    paramList = []
    resnet18 = resnet.resnet18(pretrained=True )
    for param in resnet18.parameters():
        paramList.append(param )

    cnt = 0
    for param in network.parameters():
        if paramList[cnt ].size() == param.size():
            param.data.copy_(paramList[cnt].data )
            #param.data.copy_(param.data )
            if isOutput:
                print(param.size() )
        else:
            print(param.shape, paramList[cnt].shape )
            break
        cnt += 1

